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

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.expert_cache import (
    DummyFirstKPredictor,
    EvictionPolicy,
    ExpertCacheOffloader,
    ExpertPredictor,
    LFUEviction,
    _GlobalExpertCache,
    create_eviction,
    create_predictor,
    split_hits_misses,
)

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
        cache.register_layer(name, module, {"w13_weight": _layer_cpu(layer_idx, E)})
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
    with pytest.raises(ValueError, match="Unknown expert-cache eviction policy"):
        create_eviction("mru")
    with pytest.raises(ValueError, match="waves must be 1 or 2"):
        ExpertCacheOffloader(pool_experts=8, waves=3)


def test_lfu_rank_orders_nonhorizon_low_freq_first():
    policy = LFUEviction()
    # non-horizon before horizon; within a tier, lower freq first.
    assert policy.rank(("l", 0), freq=5, in_horizon=False) < policy.rank(
        ("l", 1), freq=1, in_horizon=True
    )
    assert policy.rank(("l", 0), freq=1, in_horizon=False) < policy.rank(
        ("l", 1), freq=9, in_horizon=False
    )


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


def test_always_on_hot_never_evicted():
    off, cache, _ = _wire(E=8, pool_experts=4)
    cache.load([("layer.0", 0)], horizon=set())
    cache.pinned_hot.add(("layer.0", 0))
    # Fill + churn the rest; the hot expert must survive.
    cache.load([("layer.1", g) for g in range(1, 4)], horizon=set())
    cache.pin_step(set())
    cache.load([("layer.1", g) for g in range(4, 7)], horizon=set())
    assert cache.is_resident("layer.0", 0)


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
    victims = cache._pick_victims(2, horizon)
    assert set(victims) == {("layer.0", 0), ("layer.0", 1)}


def test_pool_exhausted_when_all_pinned_raises():
    off, cache, _ = _wire(E=4, pool_experts=4)
    cache.load([("layer.0", g) for g in range(4)], horizon=set())
    cache.pin_step({("layer.0", g) for g in range(4)})
    with pytest.raises(RuntimeError, match="pool exhausted"):
        cache.load([("layer.1", 0)], horizon=set())


def test_memory_accounting_independent_of_routing():
    off, cache, mods = _wire(E=8, pool_experts=5)
    # pool (5) + contiguous compute buffer (global_num_experts=8), each row 16 B.
    assert cache.resident_bytes() == (5 + 8) * (4 * 2 * 2)
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


def test_transfer_budget_zero_when_misses_exceed_compute_window():
    off, _, _ = _wire()
    off._t_per_expert_ms = 1.0
    off._layer_time_ema["prefill"] = None  # not warmed -> no speculation
    assert off._transfer_budget("prefill", num_tokens=100, num_misses=10) == 0
    off._layer_time_ema["prefill"] = 0.5  # 100*0.5=50ms window
    assert off._transfer_budget("prefill", num_tokens=100, num_misses=200) == 0  # miss>window
    assert off._transfer_budget("prefill", num_tokens=100, num_misses=0) > 0


def test_horizon_keys_only_upcoming_layers():
    off, _, _ = _wire(num_layers=4, predict_k=2, prefetch_horizon=2)
    keys = off._horizon_keys(cur_idx=1)  # layers 2,3
    assert keys == {("layer.2", 0), ("layer.2", 1), ("layer.3", 0), ("layer.3", 1)}
    assert off._horizon_keys(cur_idx=3) == set()  # last layer: nothing ahead


def test_prefetch_skips_already_resident_and_loads_predicted():
    off, cache, mods = _wire(
        num_layers=2, E=8, pool_experts=8, waves=1, predict_k=3, prefetch_horizon=1
    )
    # Force a positive budget.
    off._t_per_expert_ms = 0.001
    off._layer_time_ema["prefill"] = 1.0
    # Predicted for layer.1 = {0,1,2}; make {0} already resident (promote-on-hit skip).
    cache.load([("layer.1", 0)], horizon=set())
    list(off.iter_expert_waves(mods[0], torch.zeros(32, 2, dtype=torch.long, device=DEVICE)))
    assert cache.is_resident("layer.1", 1)
    assert cache.is_resident("layer.1", 2)
    assert off.prefetch_loaded == 2  # expert 0 was already resident, not re-loaded


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


@requires_cuda
@pytest.mark.parametrize("waves", [1, 2])
def test_pool_wave_sum_equals_full_residency(waves):
    """Driving the real cache: experts sit at non-identity pool slots, each wave
    gathers them into the dense compute buffer via ``publish``, and summing the
    per-wave kernel outputs equals the single full-residency output — for both
    wave modes."""
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
    # non-identity slots E..2E-1 — exercising the gather's slot lookup.
    decoy = _MoEModule("decoy", E, w1.shape[1:], w2.shape[1:])
    cache.register_layer(
        "decoy", decoy, {"w13_weight": torch.zeros_like(w1), "w2_weight": torch.zeros_like(w2)}
    )
    cache.register_layer(
        "layer.0", module, {"w13_weight": w1.clone(), "w2_weight": w2.clone()}
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
        summed += fused_experts(
            x,
            module.w13_weight.data,
            module.w2_weight.data,
            tw,
            ti,
            global_num_experts=E,
            expert_map=module._expert_map,
        ).float()

    torch.testing.assert_close(summed, reference, atol=2e-2, rtol=2e-2)
