# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the global cross-layer expert-cache offloader.

Two tiers:

* Tier A (CPU-ok) exercises the residency bookkeeping directly: hit/miss split,
  the 1- and 2-wave publish behavior, graded eviction, and the load-bearing
  safety invariant that a load (miss or prefetch) can never evict an expert the
  current step needs.
* Tier B (CUDA-only) proves the numerical claim against the real Triton kernel:
  publishing experts by pool-slot indirection and summing per-wave outputs (each
  a pass with non-wave experts masked to -1) reproduces the full-residency result
  exactly, for both waves=1 and waves=2.
"""

from collections import deque

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.offloader.expert_cache import (
    DummyFirstKPredictor,
    EvictionContext,
    ExpertCacheOffloader,
    ExpertPredictor,
    HybridExpertPredictor,
    LFUEviction,
    LRUEviction,
    MRUEviction,
    _GlobalExpertCache,
    create_eviction,
    create_predictor,
    split_hits_misses,
)
from vllm.model_executor.offloader.expert_predictor_mlp import ExpertPredictorMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class _FakeRoutedExperts(nn.Module):
    """Minimal stand-in exposing what the cache touches: `layer_name`,
    `global_num_experts`, `local_num_experts`, one target param, and buffer
    registration for `_expert_map`."""

    def __init__(self, layer_name: str, global_num_experts: int):
        super().__init__()
        self.layer_name = layer_name
        self.global_num_experts = global_num_experts
        self.local_num_experts = global_num_experts
        self.register_parameter(
            "w13_weight",
            nn.Parameter(torch.zeros(1, 2, 2, device=DEVICE), requires_grad=False),
        )


def _expert_value(layer_idx: int, gid: int) -> float:
    # Distinct, checkable per-(layer, expert) fill so we can verify a pool slot
    # holds the weights it claims to.
    return float(layer_idx * 100 + gid)


def _layer_cpu(layer_idx: int, E: int) -> torch.Tensor:
    t = torch.zeros(E, 2, 2, device=DEVICE)
    for gid in range(E):
        t[gid] = _expert_value(layer_idx, gid)
    return t


def _wire(
    num_layers: int = 2,
    E: int = 8,
    pool_experts: int = 8,
    waves: int = 2,
    predict_k: int = 0,
    prefetch_horizon: int = 0,
    predictor: str = "first_k",
) -> tuple[ExpertCacheOffloader, _GlobalExpertCache, list[_FakeRoutedExperts]]:
    """Build an offloader + populated global cache over `num_layers` fake layers,
    bypassing model discovery so `iter_expert_waves` can be driven directly."""
    off = ExpertCacheOffloader(
        pool_experts=pool_experts,
        waves=waves,
        predict_k=predict_k,
        prefetch_horizon=prefetch_horizon,
        predictor=predictor,
    )
    cache = _GlobalExpertCache(
        param_names=["w13_weight"],
        pool_experts=pool_experts,
        global_num_experts=E,
        device=DEVICE,
        eviction=off.eviction,
        per_expert_bytes=4 * 2 * 2,
    )
    modules = []
    for layer_idx in range(num_layers):
        name = f"layer.{layer_idx}"
        module = _FakeRoutedExperts(name, E)
        cache.register_layer(
            name, module, {"w13_weight": _layer_cpu(layer_idx, E)}, layer_idx
        )
        off._layer_order.append(name)
        off._layer_index[name] = layer_idx
        off._routed_experts[name] = module
        modules.append(module)
    cache.build_pool()
    off._cache = cache
    return off, cache, modules


def _assert_resident_and_correct(module, cache, layer_idx: int, gids) -> None:
    for gid in gids:
        slot = module._expert_map[gid].item()
        assert slot != -1, f"expert {gid} not resident (silent-zeroing risk)"
        assert (module.w13_weight.data[slot] == _expert_value(layer_idx, gid)).all(), (
            f"expert {gid} resident at slot {slot} holds wrong weights"
        )


def _resident_gids(module, E: int) -> list[int]:
    return [g for g in range(E) if module._expert_map[g].item() != -1]


# ---------------------------------------------------------------------------
# Tier A: hit/miss split
# ---------------------------------------------------------------------------


def test_split_hits_misses_partitions_each_expert_once():
    hits, misses = split_hits_misses({0, 1, 2}, [0, 2, 4, 5])
    assert hits == [0, 2]
    assert misses == [4, 5]
    assert sorted(hits + misses) == [0, 2, 4, 5]


def test_split_all_hits_and_all_misses():
    assert split_hits_misses({0, 1, 2}, [0, 1]) == ([0, 1], [])
    assert split_hits_misses(set(), [4, 5]) == ([], [4, 5])


# ---------------------------------------------------------------------------
# Tier A: predictor / eviction registries
# ---------------------------------------------------------------------------


def test_dummy_predictor_ignores_layer_idx_and_request():
    predictor = DummyFirstKPredictor()
    assert predictor.predict(layer_idx=0, k=4) == [0, 1, 2, 3]
    assert predictor.predict(layer_idx=17, k=4) == [0, 1, 2, 3]
    assert predictor.predict(layer_idx=0, k=0) == []


def test_predictor_registry_pluggability():
    assert isinstance(create_predictor("first_k"), DummyFirstKPredictor)

    class _IdentityPredictor(ExpertPredictor):
        def predict(self, layer_idx: int, k: int) -> list[int]:
            return [layer_idx]

    from vllm.model_executor.offloader import expert_cache

    expert_cache._PREDICTOR_REGISTRY["identity"] = _IdentityPredictor
    try:
        assert create_predictor("identity").predict(5, 1) == [5]
    finally:
        del expert_cache._PREDICTOR_REGISTRY["identity"]
    with pytest.raises(ValueError, match="Unknown expert-cache predictor"):
        create_predictor("nonexistent")


def test_eviction_registry_and_rejects_unknown_policy():
    assert isinstance(create_eviction("lfu"), LFUEviction)
    assert isinstance(create_eviction("lru"), LRUEviction)
    assert isinstance(create_eviction("mru"), MRUEviction)
    with pytest.raises(ValueError, match="Unknown expert-cache eviction policy"):
        create_eviction("arc")
    with pytest.raises(ValueError, match="waves must be 1 or 2"):
        ExpertCacheOffloader(pool_experts=8, waves=3)


def test_create_eviction_rejects_out_of_range_distance_weight():
    with pytest.raises(ValueError, match="distance_weight"):
        create_eviction("lfu", distance_weight=1.5)
    with pytest.raises(ValueError, match="distance_weight"):
        create_eviction("lru", distance_weight=-0.1)


# ---------------------------------------------------------------------------
# Tier A: hybrid predictor (Markov marginal cold-start + per-layer MLP)
# ---------------------------------------------------------------------------


def _write_hybrid_artifacts(
    tmp_path, num_layers: int = 3, num_experts: int = 8, hidden_dim: int = 4
) -> str:
    """Write a tiny markov_table.npz + one MLP checkpoint (layer 1 only),
    matching the on-disk contract `HybridExpertPredictor` expects: `<dir>/
    markov_table.npz` and `<dir>/mlp_ckpts/raw_hidden/layer_{L:02d}/
    best_model.pt`. Each layer's marginal is a distinct ascending ramp so its
    top-k is deterministic without relying on argsort tie-breaking: layer L's
    top-k is always [num_experts-1, num_experts-2, ...]."""
    marginal = np.stack(
        [np.arange(num_experts) for _ in range(num_layers)]
    ).astype(np.int64)
    np.savez_compressed(
        tmp_path / "markov_table.npz",
        counts=np.zeros((num_layers, num_experts, num_experts), dtype=np.int64),
        marginal=marginal,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=8,
    )
    ckpt_dir = tmp_path / "mlp_ckpts" / "raw_hidden" / "layer_01"
    ckpt_dir.mkdir(parents=True)
    mlp = ExpertPredictorMLP(hidden_dim, h1=16, h2=8, num_experts=num_experts)
    torch.save(
        {
            "input_dim": hidden_dim,
            "h1": 16,
            "h2": 8,
            "num_experts": num_experts,
            "dropout": 0.0,
            "state_dict": mlp.state_dict(),
        },
        ckpt_dir / "best_model.pt",
    )
    return str(tmp_path)


def test_hybrid_predictor_requires_artifacts_dir():
    with pytest.raises(ValueError, match="expert-cache-predictor-artifacts"):
        create_predictor("hybrid")


def test_hybrid_predictor_missing_markov_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="markov_table.npz"):
        create_predictor("hybrid", artifacts_dir=str(tmp_path))


def test_hybrid_predictor_missing_checkpoints_raises(tmp_path):
    np.savez_compressed(
        tmp_path / "markov_table.npz",
        counts=np.zeros((1, 8, 8), dtype=np.int64),
        marginal=np.zeros((1, 8), dtype=np.int64),
        num_layers=1,
        num_experts=8,
        topk=8,
    )
    with pytest.raises(FileNotFoundError, match="no checkpoints found"):
        create_predictor("hybrid", artifacts_dir=str(tmp_path))


def test_hybrid_predictor_layer0_is_marginal(tmp_path):
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    assert predictor.predict(0, k=3) == [7, 6, 5]


def test_hybrid_predictor_falls_back_to_marginal_without_observe(tmp_path):
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    # layer 1 has an MLP, but predict(1) before any observe() has no cached
    # hidden for layer 0 -> falls back to layer 1's own marginal.
    assert predictor.predict(1, k=2) == [7, 6]


def test_hybrid_predictor_unchecked_layer_falls_back_to_marginal(tmp_path):
    # layer 2 has no MLP checkpoint at all -> always the marginal fallback.
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_layers=3, num_experts=8)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    assert predictor.predict(2, k=2) == [7, 6]


def test_hybrid_predictor_observe_then_predict_runs_mlp(tmp_path):
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8, hidden_dim=4)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    hidden = torch.randn(5, 4, device=DEVICE)  # 5-token step
    predictor.observe(
        layer_idx=0,
        topk_ids=torch.zeros(5, 1, dtype=torch.long, device=DEVICE),
        hidden_states=hidden,
    )
    top2 = predictor.predict(1, k=2)
    assert len(top2) == 2
    assert all(0 <= g < 8 for g in top2)
    # MLP was moved to the observed hidden's device (lazy placement).
    assert next(predictor._mlps[1].parameters()).device == hidden.device


def test_hybrid_predictor_aggregates_across_step_tokens(tmp_path, monkeypatch):
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_layers=2, num_experts=4, hidden_dim=2)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)

    # Force per-token distributions to disagree, so a correct aggregation
    # (summed across tokens) is required to recover both preferred experts --
    # this would fail if predict() only looked at one token's row.
    def fake_forward(x):
        n = x.shape[0]
        out = torch.zeros(n, 4)
        out[0, 0] = 10.0  # token 0 strongly prefers expert 0
        out[1:, 1] = 10.0  # every other token strongly prefers expert 1
        return out

    monkeypatch.setattr(predictor._mlps[1], "forward", fake_forward)
    hidden = torch.zeros(5, 2)
    predictor.observe(
        layer_idx=0, topk_ids=torch.zeros(5, 1, dtype=torch.long), hidden_states=hidden
    )
    assert set(predictor.predict(1, k=2)) == {0, 1}


def test_hybrid_predictor_only_caches_latest_layer_observation(tmp_path):
    artifacts_dir = _write_hybrid_artifacts(tmp_path)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    predictor.observe(
        layer_idx=0,
        topk_ids=torch.zeros(3, 1, dtype=torch.long),
        hidden_states=torch.randn(3, 4),
    )
    predictor.observe(
        layer_idx=5,
        topk_ids=torch.zeros(3, 1, dtype=torch.long),
        hidden_states=torch.randn(3, 4),
    )
    # Bounded memory: only the most recent observation is retained regardless
    # of prefetch_horizon.
    assert set(predictor._hidden.keys()) == {5}


class _FixedLogitsMLP:
    """Deterministic stand-in for a per-layer MLP: returns pre-set logits
    per row instead of running a real trained network, so `_mlp_topk`'s
    batched-token aggregation can be tested in isolation from the model."""

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._logits


def test_mlp_topk_merges_batch_and_ranks_overlap_first():
    """Two tokens' predicted distributions must be SUMMED (not concatenated)
    before top-k: an expert both tokens agree on ("overlap") ranks ahead of
    one only a single token favors, and the result never exceeds k ids even
    though every token votes -- so a batch of N tokens yields <=k unique
    predicted experts, not N*k."""
    E, k = 8, 3
    logits = torch.zeros(2, E)
    logits[:, 5] = 10.0  # both tokens strongly favor expert 5 (the overlap)
    logits[0, 2] = 8.0  # token 0 alone also favors expert 2
    logits[1, 7] = 3.0  # token 1 alone weakly favors expert 7
    mlp = _FixedLogitsMLP(logits)
    top = HybridExpertPredictor._mlp_topk(mlp, torch.zeros(2, 4), k)
    ids = top.tolist()
    assert len(ids) <= k
    assert ids[0] == 5, "the overlapping expert must rank first"


@requires_cuda
def test_hybrid_predictor_cuda_graph_matches_eager(tmp_path):
    """The CUDA-graph fast path must be numerically identical to the eager
    forward -- graph replay recomputes the same MLP+topk, so the predicted
    expert set must match exactly."""
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8, hidden_dim=4)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    hidden = torch.randn(5, 4, device=DEVICE)
    predictor.observe(
        0, torch.zeros(5, 1, dtype=torch.long, device=DEVICE), hidden
    )

    predictor._cuda_graphs_ok = False  # force eager
    eager = predictor.predict(1, k=4)
    predictor._cuda_graphs_ok = True  # re-enable -> captures + replays a graph
    graphed = predictor.predict(1, k=4)

    assert predictor._graphs, "graph was not captured on the CUDA path"
    assert graphed == eager


@requires_cuda
def test_hybrid_predictor_graph_reused_and_tracks_new_input(tmp_path):
    """A second call at the same shape replays the cached graph (no new
    capture), and still reflects a *new* observed hidden -- i.e. the graph
    reads the refreshed static input, not a stale capture-time snapshot."""
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8, hidden_dim=4)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)

    predictor.observe(0, torch.zeros(3, 1, dtype=torch.long, device=DEVICE),
                      torch.randn(3, 4, device=DEVICE))
    first = predictor.predict(1, k=4)
    assert len(predictor._graphs) == 1

    predictor.observe(0, torch.zeros(3, 1, dtype=torch.long, device=DEVICE),
                      torch.randn(3, 4, device=DEVICE))
    second = predictor.predict(1, k=4)
    assert len(predictor._graphs) == 1  # reused, not re-captured
    # Cross-check the replay tracked the new input by comparing to eager.
    predictor._cuda_graphs_ok = False
    assert predictor.predict(1, k=4) == second
    assert isinstance(first, list) and isinstance(second, list)


@requires_cuda
def test_hybrid_predictor_large_batch_skips_graph(tmp_path):
    """Token counts above `_MAX_GRAPH_BATCH` (large prefill chunks) take the
    eager path and never capture a graph, bounding graph memory."""
    artifacts_dir = _write_hybrid_artifacts(tmp_path, num_experts=8, hidden_dim=4)
    predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    big = predictor._MAX_GRAPH_BATCH + 1
    predictor.observe(0, torch.zeros(big, 1, dtype=torch.long, device=DEVICE),
                      torch.randn(big, 4, device=DEVICE))
    out = predictor.predict(1, k=4)
    assert predictor._graphs == {}  # no graph for an oversized batch
    assert len(out) == 4


def test_hybrid_predictor_mlp_not_among_routed_experts(tmp_path):
    """Structural guard for "keep the predictor MLP resident, never
    offloaded": the MLP module must never be one of the modules the cache
    wraps for CPU-pinned offload/eviction (`_routed_experts`), since only
    those are touched by `_wrap_routed_experts`/`_CpuParamOffloader`."""
    artifacts_dir = _write_hybrid_artifacts(tmp_path)
    off, cache, mods = _wire(num_layers=2, E=8, pool_experts=8, waves=1)
    off.predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    hidden = torch.randn(2, 4, device=DEVICE)
    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1]], device=DEVICE), hidden))
    assert isinstance(off.predictor, HybridExpertPredictor)
    mlp = off.predictor._mlps[1]
    assert mlp not in off._routed_experts.values()
    assert next(mlp.parameters()).device.type == DEVICE.type  # moved via observe()


def test_iter_expert_waves_calls_observe_before_horizon(tmp_path):
    """`observe` must run before `_horizon_keys` reads the cached hidden for
    the very same step, otherwise predict(cur_idx+1) sees stale/no data."""
    artifacts_dir = _write_hybrid_artifacts(
        tmp_path, num_layers=2, num_experts=8, hidden_dim=4
    )
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, waves=1, prefetch_horizon=1
    )
    off.predictor = create_predictor("hybrid", artifacts_dir=artifacts_dir)
    hidden = torch.randn(2, 4, device=DEVICE)
    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1]], device=DEVICE), hidden))
    # observe(0, ..., hidden) ran, so predict(1) used the MLP, not the
    # layer-1 marginal fallback -> _hidden actually got populated for layer 0.
    assert 0 in off.predictor._hidden


def test_lfu_rank_orders_nonhorizon_low_freq_first():
    policy = LFUEviction()
    ctx = EvictionContext(cur_layer_idx=0, num_layers=4, tick=10, max_freq=9)
    # non-horizon before horizon; within a tier, lower freq first.
    assert policy.rank(("l", 0), 5, 0, False, 0, ctx) < policy.rank(
        ("l", 1), 1, 0, True, 0, ctx
    )
    assert policy.rank(("l", 0), 1, 0, False, 0, ctx) < policy.rank(
        ("l", 1), 9, 0, False, 0, ctx
    )


def test_lru_rank_orders_nonhorizon_stale_first():
    policy = LRUEviction()
    ctx = EvictionContext(cur_layer_idx=0, num_layers=4, tick=10, max_freq=1)
    # non-horizon before horizon; within a tier, the older (smaller last_used,
    # i.e. staler) entry goes first.
    assert policy.rank(("l", 0), 0, 2, False, 0, ctx) < policy.rank(
        ("l", 1), 0, 8, False, 0, ctx
    )
    assert policy.rank(("l", 0), 0, 9, False, 0, ctx) < policy.rank(
        ("l", 1), 0, 1, True, 0, ctx
    )


def test_mru_rank_orders_nonhorizon_fresh_first():
    """MRU is the mirror of LRU: the MOST recently used goes first."""
    policy = MRUEviction()
    ctx = EvictionContext(cur_layer_idx=0, num_layers=4, tick=10, max_freq=1)
    assert policy.rank(("l", 0), 0, 8, False, 0, ctx) < policy.rank(
        ("l", 1), 0, 2, False, 0, ctx
    )


def test_lfu_distance_weight_zero_reproduces_plain_lfu():
    """`distance_weight=0` must be a pure passthrough to plain graded-LFU --
    a regression guard for the pre-Phase-5 behavior."""
    plain = LFUEviction(distance_weight=0.0)
    ctx = EvictionContext(cur_layer_idx=10, num_layers=48, tick=100, max_freq=20)
    # layer_idx varies wildly but must have zero effect at w=0.
    r_near = plain.rank(("l", 0), 5, 0, False, 11, ctx)
    r_far = plain.rank(("l", 0), 5, 0, False, 9, ctx)
    assert r_near == r_far == (False, 5 / 20)


def test_lfu_distance_weight_prioritizes_next_layer_over_previous():
    """At layer i, the i-1 experts (just executed, furthest from next use)
    must be more evictable than the i+1 experts (about to run), at equal
    frequency -- the exact intuition requested: least priority weight for
    what was just used, most for what's coming up next."""
    policy = LFUEviction(distance_weight=1.0)
    n = 8
    ctx = EvictionContext(cur_layer_idx=4, num_layers=n, tick=10, max_freq=1)
    prev_layer_rank = policy.rank(("l", 3), 0, 0, False, 3, ctx)  # i-1
    next_layer_rank = policy.rank(("l", 5), 0, 0, False, 5, ctx)  # i+1
    assert prev_layer_rank < next_layer_rank


def test_lru_distance_weight_prioritizes_next_layer_over_previous():
    policy = LRUEviction(distance_weight=1.0)
    n = 8
    ctx = EvictionContext(cur_layer_idx=4, num_layers=n, tick=10, max_freq=1)
    prev_layer_rank = policy.rank(("l", 3), 0, 0, False, 3, ctx)
    next_layer_rank = policy.rank(("l", 5), 0, 0, False, 5, ctx)
    assert prev_layer_rank < next_layer_rank


# ---------------------------------------------------------------------------
# Tier A: single-writer residency + 1/2-wave publish
# ---------------------------------------------------------------------------


def test_waves1_single_pass_resolves_all_needed():
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    topk_ids = torch.tensor([[1, 4], [5, 6], [7, 1]], device=DEVICE)
    yielded = list(off.iter_expert_waves(mods[0], topk_ids))
    assert len(yielded) == 1
    needed = [1, 4, 5, 6, 7]
    assert sorted(_resident_gids(mods[0], 8)) == needed
    _assert_resident_and_correct(mods[0], cache, 0, needed)


def test_waves2_hits_then_misses_each_expert_once():
    off, cache, mods = _wire(E=8, pool_experts=8, waves=2)
    # Pre-resident hits {1,2}; needed {1,2,5,6} -> wave0 hits, wave1 misses.
    cache.load([("layer.0", 1), ("layer.0", 2)], horizon=set())
    topk_ids = torch.tensor([[1, 2], [5, 6]], device=DEVICE)
    seen_per_wave = []
    for _ in off.iter_expert_waves(mods[0], topk_ids):
        seen_per_wave.append(sorted(_resident_gids(mods[0], 8)))
    assert len(seen_per_wave) == 2
    assert seen_per_wave[0] == [1, 2]  # wave 0 resolves only hits
    assert seen_per_wave[1] == [5, 6]  # wave 1 resolves only misses
    # each needed expert resolved in exactly one wave
    assert sorted(seen_per_wave[0] + seen_per_wave[1]) == [1, 2, 5, 6]
    _assert_resident_and_correct(mods[0], cache, 0, [5, 6])


@requires_cuda
def test_waves2_mixed_hits_misses_times_each_wave_separately():
    """The compute-timing sample for a mixed hits+misses step under waves=2
    must bracket each wave's compute separately (one (start, end) pair per
    wave), so the wait for the miss-copy between wave 0 and wave 1 is never
    included in the measured compute time -- that wait used to get folded
    into a single start/end bracket spanning both waves, inflating
    `_layer_time_ema` by roughly 2x and making `_transfer_budget` overestimate
    available prefetch slack."""
    off, cache, mods = _wire(E=8, pool_experts=8, waves=2)
    cache.load([("layer.0", 1), ("layer.0", 2)], horizon=set())
    topk_ids = torch.tensor([[1, 2], [5, 6]], device=DEVICE)
    list(off.iter_expert_waves(mods[0], topk_ids))

    pending = off._pending_timing["layer.0"]
    assert len(pending) == 1
    _, _, events = pending[0]
    assert len(events) == 2  # one pair for wave 0 (hits), one for wave 1 (misses)


def test_waves2_no_hits_degenerates_to_single_wave():
    off, _, mods = _wire(E=8, pool_experts=8, waves=2)
    topk_ids = torch.tensor([[4, 5]], device=DEVICE)
    yielded = list(off.iter_expert_waves(mods[0], topk_ids))
    assert len(yielded) == 1
    assert sorted(_resident_gids(mods[0], 8)) == [4, 5]


# ---------------------------------------------------------------------------
# Tier A: eviction safety + graded policy + memory accounting
# ---------------------------------------------------------------------------


def test_load_never_evicts_pinned_current_step():
    # Pool holds exactly one layer. 6 pinned needs + 2 unpinned stragglers.
    off, cache, _ = _wire(E=8, pool_experts=8)
    cache.load([("layer.0", g) for g in range(6)], horizon=set())  # will pin these
    cache.load([("layer.1", 90 % 8), ("layer.1", 91 % 8)], horizon=set())  # unpinned
    # mark the 6 layer-0 experts as the current step's working set
    cache.pin_step({("layer.0", g) for g in range(6)})
    before_unpinned = {
        cache.slot_to_key[s]
        for s in range(8)
        if cache.slot_to_key[s] is not None
        and cache.slot_to_key[s][0] == "layer.1"
    }
    # need 2 new slots -> must evict, and may only take the 2 unpinned ones
    cache.load([("layer.0", 6), ("layer.0", 7)], horizon=set())
    for g in range(8):
        assert cache.is_resident("layer.0", g), f"pinned/new expert {g} was evicted"
    assert not (before_unpinned & set(cache.key_to_slot)), (
        "expected the unpinned stragglers to be the eviction victims"
    )


def test_only_current_step_is_protected_from_eviction():
    """Only `_step_pinned` (the current step's working set) restricts
    eviction now -- there is no separate "always-on" protection tier any
    more (removed per request: the only limit on eviction is the current
    step's needs; every other resident expert, including ones warmed via
    `_always_on_experts`, is fair game)."""
    off, cache, _ = _wire(E=8, pool_experts=4)
    cache.load([("layer.0", 0)], horizon=set())
    cache.load([("layer.1", g) for g in range(1, 4)], horizon=set())
    # bump everything except ("layer.0", 0) so it is uniquely least-frequent
    # and therefore the deterministic LFU eviction victim.
    for g in range(1, 4):
        cache.bump("layer.1", [g])
    cache.pin_step(set())
    cache.load([("layer.1", 4)], horizon=set())
    assert not cache.is_resident("layer.0", 0), (
        "no protection tier survives outside _step_pinned any more"
    )


def test_graded_eviction_protects_horizon_as_last_resort():
    off, cache, _ = _wire(E=8, pool_experts=4)
    # Two non-horizon (freq bumped high) + two horizon (low freq) resident.
    cache.load(
        [("layer.0", 0), ("layer.0", 1), ("layer.1", 2), ("layer.1", 3)],
        horizon=set(),
    )
    cache.freq[("layer.0", 0)] = 9
    cache.freq[("layer.0", 1)] = 9
    horizon = {("layer.1", 2), ("layer.1", 3)}
    # Need 2 victims: high-freq non-horizon should still be picked before the
    # low-freq horizon experts (horizon = last resort).
    victims = cache._pick_victims(2, horizon, cur_layer_idx=0)
    assert set(victims) == {("layer.0", 0), ("layer.0", 1)}


def test_horizon_protected_as_last_resort_under_all_policies():
    """LRU and MRU must also keep horizon experts as a last resort, not just
    LFU (the only policy this was previously tested against)."""
    for eviction in (LRUEviction(), MRUEviction()):
        off, cache, _ = _wire(E=8, pool_experts=4)
        cache.eviction = eviction
        cache.load(
            [("layer.0", 0), ("layer.0", 1), ("layer.1", 2), ("layer.1", 3)],
            horizon=set(),
        )
        # Recency-bump the non-horizon pair so they are not the "oldest"
        # candidates either -- isolates the horizon tier from recency.
        cache.bump("layer.0", [0, 1])
        horizon = {("layer.1", 2), ("layer.1", 3)}
        victims = cache._pick_victims(2, horizon, cur_layer_idx=0)
        assert set(victims) == {("layer.0", 0), ("layer.0", 1)}, eviction


def test_pool_exhausted_when_all_pinned_raises():
    off, cache, _ = _wire(E=4, pool_experts=4)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.pin_step({("layer.0", g) for g in range(4)})
    with pytest.raises(RuntimeError, match="pool exhausted"):
        cache.load([("layer.1", 0)], horizon=set())


def test_freq_is_global_and_survives_an_evict_reload_cycle():
    """`freq` is keyed globally (all layers, hit+miss via `bump`), never reset
    or deleted on eviction -- so a cold-then-hot-again expert's history isn't
    lost while it's non-resident."""
    off, cache, _ = _wire(E=8, pool_experts=4)
    cache.load([("layer.0", 0)], horizon=set())
    cache.bump("layer.0", [0])
    cache.bump("layer.0", [0])
    cache.bump("layer.0", [0])
    assert cache.freq[("layer.0", 0)] == 3

    # Fill the rest of the pool with keys bumped MORE, so ("layer.0", 0) is
    # the globally least-frequent resident -- the deterministic LFU victim
    # once one more slot is needed.
    cache.load([("layer.1", g) for g in range(3)], horizon=set())
    for g in range(3):
        for _ in range(5):
            cache.bump("layer.1", [g])
    cache.load([("layer.1", 3)], horizon=set())  # pool full -> forces 1 eviction
    assert not cache.is_resident("layer.0", 0)
    assert cache.freq[("layer.0", 0)] == 3, "freq must survive eviction, not reset"

    # Reload it -- prior history must still be there, now incremented further.
    cache.load([("layer.0", 0)], horizon=set())
    cache.bump("layer.0", [0])
    assert cache.freq[("layer.0", 0)] == 4


def test_memory_accounting_independent_of_routing():
    off, cache, mods = _wire(E=8, pool_experts=5)
    # P7.4 (structural): only the pool (5 slots) is GPU-resident, each row 16 B.
    assert cache.resident_bytes() == 5 * (4 * 2 * 2)
    before = cache.resident_bytes()
    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1, 2, 3]], device=DEVICE)))
    assert cache.resident_bytes() == before  # pool size fixed regardless of load


# ---------------------------------------------------------------------------
# Tier A: transfer budget + prefetch scheduling
# ---------------------------------------------------------------------------


def test_batch_bucket_prefill_vs_decode_without_forward_context():
    off, _, _ = _wire()
    assert off._batch_bucket(torch.zeros(1, 2, device=DEVICE)) == "decode"
    assert off._batch_bucket(torch.zeros(32, 2, device=DEVICE)) == "prefill"


def test_transfer_budget_zero_until_warmed_then_sizes_to_free_window():
    """The budget is 0 until the compute EMA warms, then sizes to the
    copy-free window (last-wave compute + gap). `num_misses` no longer
    subtracts from it -- the reactive misses are excluded by construction
    (they finish before the last wave starts), so the budget is independent
    of how many misses this step had."""
    off, _, _ = _wire()
    off._t_per_expert_ms = 1.0
    off._layer_time_ema["prefill"] = None  # not warmed -> no speculation
    assert off._transfer_budget("prefill", num_tokens=100, num_misses=10) == 0
    off._layer_time_ema["prefill"] = 0.5  # falls back to this until last-wave warms
    # A huge miss count must NOT zero the budget any more.
    many = off._transfer_budget("prefill", num_tokens=100, num_misses=200)
    none = off._transfer_budget("prefill", num_tokens=100, num_misses=0)
    assert many > 0
    assert many == none  # budget independent of num_misses


def test_transfer_budget_sizes_to_last_wave_not_total_compute():
    """Once the last-wave EMA is warm, the budget uses it (the copy-free
    window is only the last wave's compute + gap), NOT the full-step compute
    EMA -- so a small last wave caps the budget below what total compute
    would allow, even when total compute is large."""
    off, _, _ = _wire()
    off._t_per_expert_ms = 1.0
    off._layer_time_ema["prefill"] = 1.0  # large total-step compute
    off._last_wave_time_ema["prefill"] = 0.1  # but the last wave is small
    budget = off._transfer_budget("prefill", num_tokens=100, num_misses=0)
    # window = last_wave(0.1) * 100 * 0.9 safety = 9 experts, not 90.
    assert budget == 9


def test_consume_timing_tracks_last_wave_separately():
    """`_consume_timing` must feed BOTH the total-compute EMA (sum of all
    waves) and the last-wave EMA (only the final wave's pair)."""
    off, _, _ = _wire()
    wave0 = (_FakeEvent(ready=True, elapsed=3.0), _FakeEvent(ready=True))
    wave1 = (_FakeEvent(ready=True, elapsed=1.0), _FakeEvent(ready=True))
    off._pending_timing["layer.0"] = deque([("decode", 1, [wave0, wave1])])
    off._consume_timing("layer.0")
    assert off._layer_time_ema["decode"] == pytest.approx(4.0)  # 3+1 summed
    assert off._last_wave_time_ema["decode"] == pytest.approx(1.0)  # last only


def test_transfer_budget_includes_interlayer_gap_ema():
    """`_transfer_budget` must count the measured inter-layer gap (attention
    + other non-MoE work between this layer's prefetch decision and the next
    layer's need) as additional available slack, not just this layer's own
    compute time -- that gap was previously invisible/unused."""
    off, _, _ = _wire()
    off._t_per_expert_ms = 1.0
    off._layer_time_ema["prefill"] = 0.1  # 100*0.1=10ms window from compute alone
    without_gap = off._transfer_budget("prefill", num_tokens=100, num_misses=0)
    off._layer_gap_ema["prefill"] = 50.0  # a large measured inter-layer gap
    with_gap = off._transfer_budget("prefill", num_tokens=100, num_misses=0)
    assert with_gap > without_gap


def test_consume_gap_timing_retries_unready_sample_instead_of_discarding():
    off, _, _ = _wire()
    start, end = _FakeEvent(ready=False, elapsed=5.0), _FakeEvent(ready=False)
    off._pending_gap_timing.append(("decode", start, end))

    off._consume_gap_timing()
    assert off._gap_resolved_count["decode"] == 0
    assert len(off._pending_gap_timing) == 1
    assert off._layer_gap_ema["decode"] is None

    end.ready = True
    off._consume_gap_timing()
    assert off._gap_resolved_count["decode"] == 1
    assert len(off._pending_gap_timing) == 0
    assert off._layer_gap_ema["decode"] == pytest.approx(5.0)


@requires_cuda
def test_iter_expert_waves_measures_real_interlayer_gap():
    """Two consecutive `iter_expert_waves` calls (as happens once per layer
    in a real forward pass) must produce one resolved gap sample spanning
    from the first call's prefetch-issue point to the second call's entry."""
    off, cache, mods = _wire(num_layers=2, E=8, pool_experts=8, waves=1)
    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1]], device=DEVICE)))
    assert off._last_step_done_ev is not None
    list(off.iter_expert_waves(mods[1], torch.tensor([[2, 3]], device=DEVICE)))
    torch.cuda.synchronize()
    off._consume_gap_timing()
    assert off._gap_resolved_count["decode"] == 1
    assert off._layer_gap_ema["decode"] is not None
    assert off._layer_gap_ema["decode"] >= 0.0


def test_effective_transfer_rate_falls_back_to_startup_calibration():
    """Before any real transfer has been measured (`real_ms_per_expert_ema`
    is None), `_transfer_budget` must use the once-at-startup calibration."""
    off, cache, _ = _wire()
    off._t_per_expert_ms = 2.0
    assert cache.real_ms_per_expert_ema is None
    assert off._effective_t_per_expert_ms() == 2.0


def test_effective_transfer_rate_prefers_live_ema_once_measured():
    """Once real transfers have been measured, the live EMA -- not the
    once-at-startup snapshot -- must drive the budget calculation (fixes the
    measured ~16% overestimate from the once-at-startup calibration, exp12)."""
    off, cache, _ = _wire()
    off._t_per_expert_ms = 2.0  # stale/inaccurate startup calibration
    cache.real_ms_per_expert_ema = 0.5  # live-measured, much faster
    assert off._effective_t_per_expert_ms() == 0.5

    off._layer_time_ema["prefill"] = 1.0
    # 100 tokens * 1.0ms = 100ms compute; 10 misses * 0.5ms = 5ms transfer.
    # Using the stale 2.0ms/expert instead would compute t_miss=20ms and a
    # smaller budget -- confirm the live rate, not the stale one, is used.
    budget_live = off._transfer_budget("prefill", num_tokens=100, num_misses=10)
    cache.real_ms_per_expert_ema = None  # revert to the stale calibration
    budget_stale = off._transfer_budget("prefill", num_tokens=100, num_misses=10)
    assert budget_live > budget_stale


@requires_cuda
def test_drain_transfer_timing_updates_live_ema():
    """`drain_transfer_timing` must update `real_ms_per_expert_ema` (not just
    the lifetime sum/count) from a real, resolved `load()`."""
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    assert cache.real_ms_per_expert_ema is None
    cache.load([("layer.0", 3), ("layer.0", 4)], horizon=set())
    torch.cuda.synchronize()
    cache.drain_transfer_timing()
    assert cache.real_ms_per_expert_ema is not None
    assert cache.real_ms_per_expert_ema >= 0.0


# ---------------------------------------------------------------------------
# Tier A: compute-timing accuracy diagnostics (never-discard redesign)
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Deterministic stand-in for a CUDA event -- avoids racing real GPU
    timing to exercise the not-ready-yet / now-ready code paths."""

    def __init__(self, ready: bool, elapsed: float = 1.0):
        self.ready = ready
        self._elapsed = elapsed

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, _other) -> float:
        return self._elapsed


def test_consume_timing_retries_unready_sample_instead_of_discarding():
    """The redesigned `_pending_timing` must retry a not-yet-resolved sample
    on the next call rather than silently dropping it -- the bug the old
    single-slot (plain dict, overwritten each call) design had."""
    off, _, _ = _wire()
    start, end = _FakeEvent(ready=False), _FakeEvent(ready=False)
    off._pending_timing["layer.0"] = deque([("decode", 4, [(start, end)])])

    off._consume_timing("layer.0")
    assert off._timing_resolved_count["decode"] == 0
    assert len(off._pending_timing["layer.0"]) == 1  # retried, not discarded
    assert off._layer_time_ema["decode"] is None

    end.ready = True  # the "GPU work" has now completed
    off._consume_timing("layer.0")
    assert off._timing_resolved_count["decode"] == 1
    assert len(off._pending_timing["layer.0"]) == 0
    assert off._layer_time_ema["decode"] == pytest.approx(1.0 / 4)


@requires_cuda
def test_pending_timing_write_site_detects_would_have_discarded():
    """If a prior compute-timing sample for a layer is still unresolved when
    a new one is about to be recorded, the OLD single-slot design would have
    silently overwritten (discarded) it. Verify the new design instead
    retries the old sample AND counts the near-miss."""
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    stuck_start, stuck_end = _FakeEvent(ready=False), _FakeEvent(ready=False)
    off._pending_timing["layer.0"] = deque([("decode", 1, [(stuck_start, stuck_end)])])

    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1]], device=DEVICE)))

    pending = off._pending_timing["layer.0"]
    # not discarded
    assert any(stuck_end in (e for _, e in entry[2]) for entry in pending)
    assert off._timing_would_have_discarded_count["decode"] == 1


@requires_cuda
def test_drain_transfer_timing_accumulates_real_load_time():
    """A real `load()` call's returned event, once resolved, must feed
    `real_transfer_ms_sum`/`real_transfer_expert_count` -- the live
    counterpart to the once-at-startup PCIe calibration."""
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    assert cache.real_transfer_expert_count == 0
    ev = cache.load([("layer.0", 3), ("layer.0", 4)], horizon=set())
    assert ev is not None
    torch.cuda.synchronize()  # let the real H2D copy actually complete
    cache.drain_transfer_timing()
    assert cache.real_transfer_expert_count == 2
    assert cache.real_transfer_ms_sum >= 0.0


def test_timing_diagnostics_exposed_in_get_stats():
    off, _, _ = _wire()
    off._pending_timing["layer.0"] = deque(
        [("decode", 2, [(_FakeEvent(ready=True, elapsed=2.0), _FakeEvent(ready=True))])]
    )
    off._consume_timing("layer.0")
    stats = off.get_stats()
    assert stats["decode"]["resolved"] == 1
    assert stats["decode"]["would_have_discarded"] == 0
    assert stats["decode"]["current_ema_ms_per_token"] == pytest.approx(1.0)
    assert "transfer_calibration" in stats
    assert stats["transfer_calibration"]["real_sample_experts"] == 0


def test_horizon_keys_only_upcoming_layers():
    off, _, _ = _wire(num_layers=4, predict_k=2, prefetch_horizon=2)
    keys = off._horizon_keys(cur_idx=1)  # layers 2,3
    assert keys == {("layer.2", 0), ("layer.2", 1), ("layer.3", 0), ("layer.3", 1)}
    # Cyclic: the last layer's horizon wraps into the *next* forward pass's
    # early layers (decode runs layer N-1 then layer 0 back-to-back, so this
    # is a real prefetch opportunity, not a bug).
    assert off._horizon_keys(cur_idx=3) == {
        ("layer.0", 0),
        ("layer.0", 1),
        ("layer.1", 0),
        ("layer.1", 1),
    }


def test_horizon_keys_wraps_cyclically_from_last_layer():
    off, _, _ = _wire(num_layers=3, predict_k=1, prefetch_horizon=1)
    # cur_idx=2 is the last layer; horizon of 1 wraps to layer 0 of the next pass.
    assert off._horizon_keys(cur_idx=2) == {("layer.0", 0)}


def test_horizon_keys_caps_at_num_layers_minus_one_never_revisits_cur_idx():
    off, _, _ = _wire(num_layers=3, predict_k=1, prefetch_horizon=10)
    # prefetch_horizon (10) far exceeds num_layers-1 (2); must cap rather than
    # wrap all the way back around to predicting cur_idx itself.
    assert off._upcoming_layers(cur_idx=0) == [1, 2]
    assert off._horizon_keys(cur_idx=0) == {("layer.1", 0), ("layer.2", 0)}


def test_upcoming_layers_empty_for_single_layer_model():
    off, _, _ = _wire(num_layers=1, predict_k=1, prefetch_horizon=1)
    assert off._upcoming_layers(cur_idx=0) == []


def test_prefetch_skips_already_resident_and_loads_predicted():
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, waves=1, predict_k=3, prefetch_horizon=1
    )
    # Force a positive budget.
    off._t_per_expert_ms = 0.001
    off._layer_time_ema["decode"] = 1.0
    # Predicted for layer.1 = {0,1,2}; make {0} already resident (promote-on-hit skip).
    cache.load([("layer.1", 0)], horizon=set())
    # Single-token (decode) step: P7.3 bypasses predict_k-based prediction
    # entirely for "prefill" steps (see test_prefetch_loaded_bucketed_by_
    # triggering_step for that behavior instead), and this test is
    # specifically about predict_k-driven prediction + promote-on-hit skip.
    list(off.iter_expert_waves(mods[0], torch.zeros(1, 2, dtype=torch.long, device=DEVICE)))
    assert cache.is_resident("layer.1", 1)
    assert cache.is_resident("layer.1", 2)
    assert off.prefetch_loaded == 2  # expert 0 was already resident, not re-loaded


def test_prefetch_budget_zero_counted(monkeypatch):
    """`_transfer_budget` returning 0 (no slack) is a distinct diagnostic
    bucket from either candidates-ran-out or budget-hit-the-ceiling -- no
    candidates were even considered."""
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, predict_k=3, prefetch_horizon=1
    )
    monkeypatch.setattr(off, "_transfer_budget", lambda *a, **k: 0)
    off._issue_prefetch(
        "decode",
        num_tokens=1,
        num_misses=0,
        predicted=[("layer.1", 0)],
        cur_layer_idx=0,
    )
    assert off.prefetch_budget_zero["decode"] == 1
    assert off.prefetch_queue_exhausted["decode"] == 0
    assert off.prefetch_budget_limited["decode"] == 0
    assert off.get_stats()["decode"]["budget_zero"] == 1


def test_prefetch_queue_exhausted_when_candidates_run_out_first(monkeypatch):
    """Budget far exceeds the candidate count -> the queue empties before the
    budget ceiling is ever hit. More budget would NOT have changed the
    outcome this step."""
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, predict_k=3, prefetch_horizon=1
    )
    monkeypatch.setattr(off, "_transfer_budget", lambda *a, **k: 100)
    predicted = [("layer.1", 0), ("layer.1", 1)]
    off._issue_prefetch(
        "decode", num_tokens=1, num_misses=0, predicted=predicted, cur_layer_idx=0
    )
    assert off.prefetch_queue_exhausted["decode"] == 1
    assert off.prefetch_budget_limited["decode"] == 0
    assert off.prefetch_loaded == 2
    stats = off.get_stats()["decode"]
    assert stats["queue_exhausted"] == 1
    assert stats["avg_budget"] == 100.0


def test_prefetch_budget_limited_when_ceiling_hit_first(monkeypatch):
    """Budget is smaller than the candidate count -> the ceiling is hit while
    candidates remain. This is the "more budget would help" signal."""
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, predict_k=3, prefetch_horizon=1
    )
    monkeypatch.setattr(off, "_transfer_budget", lambda *a, **k: 1)
    predicted = [("layer.1", 0), ("layer.1", 1), ("layer.1", 2)]
    off._issue_prefetch(
        "decode", num_tokens=1, num_misses=0, predicted=predicted, cur_layer_idx=0
    )
    assert off.prefetch_budget_limited["decode"] == 1
    assert off.prefetch_queue_exhausted["decode"] == 0
    assert off.prefetch_loaded == 1  # only 1 of 3 candidates got through


# ---------------------------------------------------------------------------
# Tier A: bucketed hit-rate / prefetch-ratio instrumentation
# ---------------------------------------------------------------------------


def test_bucketed_hits_misses_land_in_correct_bucket():
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    cache.load([("layer.0", 0)], horizon=set())  # pre-resident hit
    # decode step: single token, needs {0 (hit), 1 (miss)}.
    list(off.iter_expert_waves(mods[0], torch.tensor([[0, 1]], device=DEVICE)))
    assert off.bucket_stats["decode"] == {"hits": 1, "misses": 1, "prefetch_loaded": 0}
    assert off.bucket_stats["prefill"] == {"hits": 0, "misses": 0, "prefetch_loaded": 0}
    # prefill step (many tokens): needs {2,3,4} (all misses).
    list(
        off.iter_expert_waves(
            mods[0], torch.tensor([[2], [3], [4]] * 8, device=DEVICE)
        )
    )
    assert off.bucket_stats["prefill"]["misses"] == 3
    assert off.bucket_stats["decode"] == {"hits": 1, "misses": 1, "prefetch_loaded": 0}


def test_get_stats_bucket_hit_rate_and_prefetch_ratio():
    off, cache, mods = _wire(E=8, pool_experts=8, waves=1)
    off.bucket_stats["prefill"] = {"hits": 3, "misses": 1, "prefetch_loaded": 4}
    stats = off.get_stats()
    pf = stats["prefill"]
    assert pf["hit_rate"] == pytest.approx(3 / 4)  # 3 hits / (3 hits + 1 miss)
    assert pf["prefetch_ratio"] == pytest.approx(4 / 5)  # 4 prefetch / (1 miss + 4 prefetch)
    assert stats["decode"]["hit_rate"] == 0.0  # no traffic yet -> 0, not NaN/error


def test_prefetch_loaded_bucketed_by_triggering_step():
    # pool_experts > E: P7.3 makes a "prefill" step prefetch *every* expert of
    # the upcoming layer (not just `predict_k`), so the pool needs room for
    # that full set (E) plus layer.0's own single resident expert.
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=16, waves=1, predict_k=3, prefetch_horizon=1
    )
    off._t_per_expert_ms = 0.001
    off._layer_time_ema["prefill"] = 1.0
    # A 32-token step (bucketed "prefill" by the no-forward-context heuristic)
    # triggers the prefetch; the loaded experts must be attributed to "prefill".
    list(off.iter_expert_waves(mods[0], torch.zeros(32, 2, dtype=torch.long, device=DEVICE)))
    assert off.bucket_stats["prefill"]["prefetch_loaded"] == off.prefetch_loaded
    assert off.bucket_stats["decode"]["prefetch_loaded"] == 0


def test_predict_called_once_per_step_even_when_prefetch_fires():
    """The predictor is queried exactly once per (layer, step): the
    eviction-protection horizon and the prefetch queue share that one call,
    not one each. Matters for a costly predictor (e.g. the hybrid MLP whose
    predict() forces a GPU sync) -- a second call would double its overhead."""

    class _CountingPredictor(ExpertPredictor):
        def __init__(self):
            self.calls = 0

        def predict(self, layer_idx: int, k: int) -> list[int]:
            self.calls += 1
            return list(range(k))

    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, waves=1, predict_k=3, prefetch_horizon=1
    )
    off.predictor = _CountingPredictor()
    # Force a positive transfer budget so _issue_prefetch actually runs its
    # queue-building path (the second historical predict() call site).
    off._t_per_expert_ms = 0.001
    off._layer_time_ema["decode"] = 1.0
    # Single-token (decode) step: P7.3 bypasses the predictor entirely for
    # "prefill" steps, which would make `calls` stay 0 -- this test is
    # specifically about the predictor being queried at most once per step.
    list(off.iter_expert_waves(mods[0], torch.zeros(1, 2, dtype=torch.long, device=DEVICE)))
    # prefetch_horizon=1 -> exactly one upcoming layer -> exactly one predict().
    assert off.predictor.calls == 1


def test_periodic_stats_log_fires_only_on_layer_zero_at_interval(monkeypatch):
    off, cache, mods = _wire(num_layers=3, E=8, pool_experts=8, waves=1)
    off._STATS_LOG_INTERVAL = 2
    calls = []
    monkeypatch.setattr(off, "_log_stats_line", lambda: calls.append(off._pass_count))
    topk = torch.tensor([[0, 1]], device=DEVICE)
    for _ in range(2):  # two full passes over all 3 layers
        for m in mods:
            list(off.iter_expert_waves(m, topk))
    assert off._pass_count == 2  # only layer-0 calls increment the pass counter
    assert calls == [2]  # logged once, at pass 2 (the interval)


# ---------------------------------------------------------------------------
# Tier B: numerical equality via the real cache.publish path (CUDA)
# ---------------------------------------------------------------------------


class _MoEModule(nn.Module):
    """Fake routed-experts holding both fused-MoE weight params on device."""

    def __init__(self, layer_name, E, w1_shape, w2_shape):
        super().__init__()
        self.layer_name = layer_name
        self.global_num_experts = E
        self.local_num_experts = E
        self.register_parameter(
            "w13_weight",
            nn.Parameter(torch.zeros(1, *w1_shape, device=DEVICE), requires_grad=False),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(torch.zeros(1, *w2_shape, device=DEVICE), requires_grad=False),
        )


def _run_moe_production_path(
    x, w1, w2, tw, ti, global_num_experts, expert_map, top_k
):
    """Replicates `TritonExperts.apply()`'s actual assignment path (the one
    `ExpertCacheOffloader` is validated against via `_validate_kernel_support`)
    -- `_prepare_expert_assignment(..., ignore_invalid_experts=False)`, the
    default -- rather than going through the top-level `fused_experts()`
    convenience wrapper, which hardcodes `ignore_invalid_experts=True`. That
    flag controls two DIFFERENT low-level code paths for resolving
    `expert_map`: `True` passes it straight into the `moe_align_block_size`
    CUDA op, which (empirically verified) assumes a dense/small local index
    range and silently produces garbage for arbitrary local indices (e.g. our
    scattered pool slots); `False` (production's default) resolves it via a
    plain `expert_ids = expert_map[expert_ids]` gather afterward, which is
    correct for ANY local index. Using the wrong one here would validate a
    kernel path this offloader never actually exercises."""
    import triton.language as tl

    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _get_config_dtype_str,
        _prepare_expert_assignment,
        dispatch_fused_moe_kernel,
        try_get_optimal_moe_config,
    )

    num_tokens = x.shape[0]
    N, K = w1.shape[1], w2.shape[1]
    inter = N // 2
    dtype = x.dtype
    compute_type = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, dtype=dtype
    )
    config = try_get_optimal_moe_config(
        w1.size(), w2.size(), top_k, config_dtype, num_tokens, block_shape=None
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        ti, config, num_tokens, top_k, global_num_experts, expert_map,
        ignore_invalid_experts=False,
    )

    cache13 = torch.empty(
        num_tokens * top_k * max(N, K), device=x.device, dtype=dtype
    )
    intermediate_cache1 = cache13[: num_tokens * top_k * N].view(num_tokens, top_k, N)
    intermediate_cache3 = cache13[: num_tokens * top_k * K].view(num_tokens, top_k, K)
    intermediate_cache2 = torch.empty(
        (num_tokens * top_k, inter), device=x.device, dtype=dtype
    )

    dispatch_fused_moe_kernel(
        x, w1, intermediate_cache1, None, None, None, tw,
        sorted_token_ids, expert_ids, num_tokens_post_padded, False, top_k, config,
        compute_type=compute_type, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
        block_shape=None, B_bias=None,
    )
    act_out = (
        F.silu(intermediate_cache1[..., :inter]) * intermediate_cache1[..., inter:]
    )
    intermediate_cache2.copy_(act_out.reshape(-1, inter))

    dispatch_fused_moe_kernel(
        intermediate_cache2, w2, intermediate_cache3, None, None, None, tw,
        sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, config,
        compute_type=compute_type, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
        block_shape=None, B_bias=None,
    )
    return intermediate_cache3.sum(dim=1)


@requires_cuda
@pytest.mark.parametrize("waves", [1, 2])
def test_pool_wave_sum_equals_full_residency(waves):
    """Driving the real cache: experts sit at non-identity, scattered pool
    slots; each wave's `publish()` points params at the pool directly and
    resolves `_expert_map` to those slots (P7.4 structural -- no gather), and
    summing the per-wave kernel outputs (via the production assignment path,
    see `_run_moe_production_path`) equals the single full-residency output
    -- for both wave modes."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    torch.manual_seed(0)
    num_tokens, hidden, inter = 64, 128, 128
    E, top_k, pool_experts = 8, 2, 20
    dtype = torch.float16

    x = torch.randn(num_tokens, hidden, device=DEVICE, dtype=dtype)
    w1 = torch.randn(E, 2 * inter, hidden, device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn(E, hidden, inter, device=DEVICE, dtype=dtype) / 10
    logits = torch.randn(num_tokens, E, device=DEVICE, dtype=torch.float32)
    tw, ti = torch.topk(torch.softmax(logits, dim=-1), top_k)
    tw, ti = tw.to(dtype), ti.to(torch.int32)

    reference = fused_experts(x, w1, w2, tw, ti, global_num_experts=E).float()

    module = _MoEModule("layer.0", E, w1.shape[1:], w2.shape[1:])
    cache = _GlobalExpertCache(
        param_names=["w13_weight", "w2_weight"],
        pool_experts=pool_experts,
        global_num_experts=E,
        device=DEVICE,
        eviction=LFUEviction(),
        per_expert_bytes=(w1[0].numel() + w2[0].numel()) * 2,
    )
    # A decoy first layer occupies slots 0..E-1 so layer.0's experts land at
    # non-identity slots E..2E-1 — exercising the pool-slot indirection.
    decoy = _MoEModule("decoy", E, w1.shape[1:], w2.shape[1:])
    cache.register_layer(
        "decoy",
        decoy,
        {"w13_weight": torch.zeros_like(w1), "w2_weight": torch.zeros_like(w2)},
        0,
    )
    cache.register_layer(
        "layer.0", module, {"w13_weight": w1.clone(), "w2_weight": w2.clone()}, 1
    )
    cache.build_pool()
    cache.load([("decoy", g) for g in range(E)], horizon=set())
    cache.load([("layer.0", g) for g in range(E)], horizon=set())
    assert cache.key_to_slot[("layer.0", 0)] != 0  # genuinely scattered

    needed = list(range(E))
    wave_groups = [needed] if waves == 1 else [needed[: E // 2], needed[E // 2 :]]

    summed = torch.zeros_like(reference)
    for group in wave_groups:
        cache.publish("layer.0", group)
        torch.cuda.synchronize()
        summed += _run_moe_production_path(
            x,
            module.w13_weight.data,
            module.w2_weight.data,
            tw,
            ti,
            E,
            module._expert_map,
            top_k,
        ).float()

    torch.testing.assert_close(summed, reference, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Phase 7: P7.3 (prefill stream-ahead), P7.5 (fixed+marginal transfer fit)
# ---------------------------------------------------------------------------


def test_prefill_bypasses_predictor_and_uses_full_expert_range():
    """P7.3: a "prefill"-bucketed step must not call the predictor at all
    (prediction is worthless when a chunk touches ~every expert anyway, and
    the MLP forward is pure waste on a large token count) and must instead
    treat every expert of the upcoming layer as a horizon/prefetch
    candidate."""

    class _CountingPredictor(ExpertPredictor):
        def __init__(self):
            self.calls = 0

        def predict(self, layer_idx: int, k: int) -> list[int]:
            self.calls += 1
            return list(range(k))

    off, cache, mods = _wire(
        num_layers=2, E=4, pool_experts=8, waves=1, predict_k=1, prefetch_horizon=1
    )
    off.predictor = _CountingPredictor()
    off._t_per_expert_ms = 0.001
    off._layer_time_ema["prefill"] = 1.0
    # 32 tokens with no forward context -> bucketed "prefill".
    topk_ids = torch.zeros(32, 2, dtype=torch.long, device=DEVICE)
    list(off.iter_expert_waves(mods[0], topk_ids))
    assert off.predictor.calls == 0, "predictor must be bypassed during prefill"
    # All E=4 experts of layer.1 (not just predict_k=1) must be horizon/
    # prefetch candidates -- verified indirectly via residency after a
    # prefetch-permitting budget.
    for g in range(4):
        assert cache.is_resident("layer.1", g), f"expert {g} should be prefetched"


def test_prefill_full_range_respects_predict_k_zero_disable():
    """`predict_k=0` (prediction explicitly disabled) must still mean "predict
    nothing" during prefill, not "use the full range" -- P7.3's bypass is
    gated on this. The predictor is still queried (with `k=0`, same as any
    other disabled-prediction step) rather than skipped outright -- disabling
    prediction means an empty *result*, not a special no-call path."""

    class _CountingPredictor(ExpertPredictor):
        def __init__(self):
            self.calls = 0

        def predict(self, layer_idx: int, k: int) -> list[int]:
            self.calls += 1
            return list(range(k))

    off, cache, mods = _wire(
        num_layers=2, E=4, pool_experts=8, waves=1, predict_k=0, prefetch_horizon=1
    )
    off.predictor = _CountingPredictor()
    predicted = off._predict_upcoming(cur_idx=0, k=0, bucket="prefill")
    assert predicted == []
    assert off.predictor.calls == 1


def test_transfer_regression_fits_fixed_plus_marginal():
    """P7.5: fitting `elapsed = fixed + marginal * n` over varying-size
    samples should recover a marginal rate close to the true per-expert cost
    when a fixed per-batch overhead is present, whereas plain `elapsed / n`
    (on a small batch) would overestimate it."""
    off, cache, _ = _wire()
    true_fixed, true_marginal = 0.2, 0.05
    # Deterministic samples at a few different batch sizes; every sample sees
    # a fixed cost plus a per-expert marginal cost.
    for n in [1, 2, 3, 4, 5, 6, 7, 8] * 4:
        elapsed = true_fixed + true_marginal * n
        cache._update_transfer_regression(n, elapsed)
    assert cache._transfer_marginal_ms_ema is not None
    assert cache._transfer_marginal_ms_ema == pytest.approx(true_marginal, rel=0.05)
    assert cache._transfer_fixed_ms_ema == pytest.approx(true_fixed, rel=0.05)
    # Plain elapsed/n on a small batch overestimates the marginal rate --
    # confirms the fit is actually doing something different (and better).
    naive_small_batch = (true_fixed + true_marginal * 2) / 2
    assert naive_small_batch > cache._transfer_marginal_ms_ema


def test_transfer_regression_ignored_until_enough_samples():
    """Before enough (varied) samples have been seen, the fit must not
    produce a value -- `_effective_t_per_expert_ms` should keep falling back
    to `real_ms_per_expert_ema`/the startup calibration rather than act on a
    noisy/ill-conditioned regression."""
    off, cache, _ = _wire()
    assert cache._transfer_marginal_ms_ema is None
    cache._update_transfer_regression(3, 1.0)
    assert cache._transfer_marginal_ms_ema is None  # only one sample so far


def test_effective_transfer_rate_prefers_fitted_marginal_once_available():
    """Once the P7.5 regression has a value, it must take priority over the
    plain `real_ms_per_expert_ema` in `_effective_t_per_expert_ms`."""
    off, cache, _ = _wire()
    cache.real_ms_per_expert_ema = 0.9
    cache._transfer_marginal_ms_ema = 0.4
    assert off._effective_t_per_expert_ms() == 0.4


# ---------------------------------------------------------------------------
# Phase 7 (round 2): P7.5 fix (variance gate), P7.8 (per-bucket eviction),
# P7.4 structural (direct pool publish), P7.12 (persistent _expert_map buffer)
# ---------------------------------------------------------------------------


def test_transfer_regression_rejects_low_variance_window():
    """P7.5 fix: a window where `n` barely varies (here 5/6, coefficient of
    variation ~0.09, well under `_XFER_MIN_CV=0.15`) must never be trusted,
    even with many samples and a technically nonzero `denom` -- this is
    exactly the "merely noisy, not exactly degenerate" case the original
    `abs(denom) < 1e-6` guard didn't catch (measured in production: fitted
    marginal swinging 2x+ between consecutive samples of the same run)."""
    off, cache, _ = _wire()
    for n in [5, 6] * 30:
        cache._update_transfer_regression(n, 0.2 + 0.05 * n)
    assert cache._transfer_marginal_ms_ema is None


def test_transfer_regression_requires_dedicated_min_effective_samples():
    """The slower, dedicated regression decay (`_XFER_REGRESSION_ALPHA`) means
    the fit must not fire off just 2 effective samples (the old threshold) --
    it needs `_XFER_MIN_EFFECTIVE_SAMPLES` of the SLOWER-decaying weight."""
    off, cache, _ = _wire()
    cache._update_transfer_regression(2, 0.3)
    cache._update_transfer_regression(8, 0.6)
    assert cache._xfer_ew_w < cache._XFER_MIN_EFFECTIVE_SAMPLES
    assert cache._transfer_marginal_ms_ema is None


def test_eviction_prefill_defaults_to_none_reuses_main_policy():
    """`eviction_prefill=None` (the default) must resolve to the same policy
    object for both buckets -- no behavior change for existing configs that
    don't opt into P7.8."""
    cache = _GlobalExpertCache(
        param_names=["w13_weight"],
        pool_experts=4,
        global_num_experts=4,
        device=DEVICE,
        eviction=LRUEviction(),
        per_expert_bytes=16,
    )
    assert cache._eviction_for("decode") is cache.eviction
    assert cache._eviction_for("prefill") is cache.eviction


def test_bucket_selects_distinct_eviction_policy():
    """P7.8: with an explicit `eviction_prefill`, a "prefill"-bucketed
    `_pick_victims` call must use it instead of the default `eviction`, and a
    "decode"-bucketed call must still use the default -- LRU and MRU pick
    opposite victims from the same recency state, so this is directly
    observable."""
    off, cache, _ = _wire(E=8, pool_experts=4)
    cache.eviction = LRUEviction()
    cache.eviction_prefill = MRUEviction()
    keys = [("layer.0", 0), ("layer.0", 1), ("layer.1", 2), ("layer.1", 3)]
    cache.load(keys, horizon=set())
    # `load()` itself doesn't bump recency -- establish it explicitly,
    # oldest -> newest: 0, 1, 2, 3.
    for layer, gid in keys:
        cache.bump(layer, [gid])
    decode_victims = cache._pick_victims(1, set(), cur_layer_idx=0, bucket="decode")
    prefill_victims = cache._pick_victims(
        1, set(), cur_layer_idx=0, bucket="prefill"
    )
    assert decode_victims == [("layer.0", 0)]  # LRU: oldest first
    assert prefill_victims == [("layer.1", 3)]  # MRU: newest first


def test_load_bucket_argument_reaches_eviction_policy():
    """`load(..., bucket=...)` must thread through to `_pick_victims` -- not
    just `_pick_victims` called directly, since that's what
    `iter_expert_waves`/`_issue_prefetch` actually call."""
    off, cache, _ = _wire(E=8, pool_experts=4)
    cache.eviction = LRUEviction()
    cache.eviction_prefill = MRUEviction()
    keys = [("layer.0", 0), ("layer.0", 1), ("layer.1", 2), ("layer.1", 3)]
    cache.load(keys, horizon=set())
    for layer, gid in keys:
        cache.bump(layer, [gid])
    cache.load([("layer.0", 4)], horizon=set(), bucket="prefill")
    # MRU (prefill) should have evicted the newest (("layer.1", 3)), not the
    # oldest (("layer.0", 0), which LRU/decode would have picked).
    assert not cache.is_resident("layer.1", 3)
    assert cache.is_resident("layer.0", 0)


def test_offloader_builds_eviction_prefill_from_config():
    """`ExpertCacheOffloader` must build a distinct `eviction_prefill` object
    only when `eviction_policy_prefill` is explicitly set, honoring a
    distance-weight override when given and falling back to the main
    `eviction_distance_weight` otherwise."""
    off_default = ExpertCacheOffloader(pool_experts=4)
    assert off_default.eviction_prefill is None

    off_explicit = ExpertCacheOffloader(
        pool_experts=4,
        eviction_policy="lru",
        eviction_policy_prefill="mru",
    )
    assert isinstance(off_explicit.eviction_prefill, MRUEviction)

    off_weighted = ExpertCacheOffloader(
        pool_experts=4,
        eviction_policy="lru",
        eviction_distance_weight=0.2,
        eviction_policy_prefill="lru",
        eviction_distance_weight_prefill=0.7,
    )
    assert off_weighted.eviction.distance_weight == 0.2
    assert off_weighted.eviction_prefill.distance_weight == 0.7

    off_inherited = ExpertCacheOffloader(
        pool_experts=4,
        eviction_policy="lru",
        eviction_distance_weight=0.3,
        eviction_policy_prefill="lru",
    )
    assert off_inherited.eviction_prefill.distance_weight == 0.3


def test_publish_local_num_experts_is_pool_size_not_wave_size():
    """P7.4 (structural): `local_num_experts` must be the constant pool size,
    not the (varying, small) number of experts this wave actually needs --
    the kernel indexes via `_expert_map`, not a dense wave-sized range."""
    off, cache, mods = _wire(E=4, pool_experts=8)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.publish("layer.0", [0, 1])
    assert mods[0].local_num_experts == 8


def test_publish_points_param_directly_at_pool_no_gather():
    """P7.4 (structural): `param.data` must alias the shared pool tensor
    itself (same storage), not a separate gathered/copied buffer."""
    off, cache, mods = _wire(E=4, pool_experts=8)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.publish("layer.0", [0, 1])
    assert mods[0].w13_weight.data.data_ptr() == cache.pool["w13_weight"].data_ptr()
    assert mods[0].w13_weight.data.shape[0] == cache.pool_experts


def test_expert_map_buffer_reused_across_publishes():
    """P7.12: the `_expert_map` buffer object must be the SAME tensor across
    repeated `publish()` calls for a layer (refilled in place), not freshly
    allocated and re-registered every call."""
    off, cache, mods = _wire(E=4, pool_experts=8)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.publish("layer.0", [0, 1])
    first = mods[0]._expert_map
    cache.publish("layer.0", [2, 3])
    second = mods[0]._expert_map
    assert first is second
    assert first is cache._expert_map_buf["layer.0"]


def test_expert_map_buffer_correctly_refilled_between_publishes():
    """The in-place refill (P7.12) must not leak a stale mapping from a
    previous `publish()` call -- an expert resident this wave but NOT in the
    new `gids` must read back as -1 (non-resident), not its old slot."""
    off, cache, mods = _wire(E=4, pool_experts=8)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.publish("layer.0", [0, 1])
    assert mods[0]._expert_map[0].item() != -1
    cache.publish("layer.0", [2, 3])
    assert mods[0]._expert_map[0].item() == -1
    assert mods[0]._expert_map[2].item() != -1
