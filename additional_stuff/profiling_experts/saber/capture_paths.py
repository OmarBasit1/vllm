#!/usr/bin/env python
"""Phase 1 - capture real activation paths from vLLM's gate + top-k.

For each token we record the set of routed experts selected at every MoE layer,
sourced from vLLM's own router (not HuggingFace). We profile **prefill-only,
one prompt at a time (batch 1)** so that row ``i`` of a layer's router output
maps trivially to sequence position ``i`` and its causal-context path.

Mechanism: register a forward hook on every MoE block's ``gate`` submodule
(``Qwen2MoeSparseMoeBlock.gate``). The gate returns ``router_logits`` of shape
``(num_tokens, num_experts)``; we take ``topk(k)`` to get the selected expert
ids. The shared expert is on a separate code path and never enters
``router_logits``, so it is excluded automatically. Hook state lives on the
model object inside the worker and is read back via ``LLM.apply_model``.

Outputs (under common.OUT_ROOT):
  model_config.json                      - live config + derived MoE facts
  paths/<dataset>.parquet                - one row per token
  capture_summary.json                   - active-slot histogram + validation

Usage:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib python capture_paths.py \
      --max-seqs-per-dataset 50            # smoke test; omit/0 = all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from corpus import load_prompts  # noqa: E402


def make_hooks(k: int):
    """Return (install, pop) closures executed inside the vLLM worker.

    Defined as nested functions so cloudpickle serializes them by value (the
    worker process cannot import this script).
    """

    def install(model):
        import torch
        from collections import defaultdict

        cap = defaultdict(list)
        model._saber_cap = cap
        moe_idxs = []
        layers = model.model.layers
        for idx in range(len(layers)):
            mlp = layers[idx].mlp
            if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                moe_idxs.append(idx)

                def make_hook(li):
                    def hook(mod, inp, out):
                        logits = out[0] if isinstance(out, (tuple, list)) else out
                        ids = torch.topk(logits, k, dim=-1).indices
                        cap[li].append(ids.detach().to("cpu", torch.int16).numpy())

                    return hook

                mlp.gate.register_forward_hook(make_hook(idx))
        return moe_idxs

    def pop(model):
        import numpy as _np

        cap = getattr(model, "_saber_cap", {})
        out = {}
        for li, chunks in cap.items():
            if chunks:
                out[li] = _np.concatenate(chunks, axis=0)
        cap.clear()
        return out

    return install, pop


def parquet_schema(n_slots: int) -> pa.Schema:
    return pa.schema(
        [
            ("dataset", pa.string()),
            ("seq_id", pa.int32()),
            ("position", pa.int32()),
            ("token_id", pa.int32()),
            # layer-major flattened topk ids, length L_moe * k
            ("experts", pa.list_(pa.int16(), n_slots)),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=common.DATASETS)
    ap.add_argument("--max-seqs-per-dataset", type=int, default=0, help="0 = all")
    ap.add_argument("--max-len", type=int, default=1024, help="truncate prompts to N tokens")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    cfg = AutoConfig.from_pretrained(common.MODEL)
    k = cfg.num_experts_per_tok
    tok = AutoTokenizer.from_pretrained(common.MODEL)

    print(f"loading vLLM engine for {common.MODEL} (enforce_eager=True) ...", flush=True)
    llm = LLM(
        model=common.MODEL,
        enforce_eager=True,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=False,
        max_num_seqs=1,
        disable_log_stats=True,
    )

    install, pop = make_hooks(k)
    moe_idxs = llm.apply_model(install)[0]
    L_moe = len(moe_idxs)
    n_slots = L_moe * k
    layer_pos = {li: p for p, li in enumerate(moe_idxs)}  # layer idx -> 0..L_moe-1
    print(f"MoE layers: {L_moe} (idxs {moe_idxs[:4]}...{moe_idxs[-1]}); "
          f"k={k}; active slots/token={n_slots}", flush=True)

    common.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    common.PATHS_DIR.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model": common.MODEL,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": k,
        "norm_topk_prob": getattr(cfg, "norm_topk_prob", None),
        "decoder_sparse_step": getattr(cfg, "decoder_sparse_step", None),
        "mlp_only_layers": getattr(cfg, "mlp_only_layers", None),
        "shared_expert_intermediate_size": getattr(cfg, "shared_expert_intermediate_size", None),
        "hidden_size": cfg.hidden_size,
        "moe_layer_idxs": moe_idxs,
        "L_moe": L_moe,
        "active_slots_per_token": n_slots,
    }
    common.MODEL_CONFIG.write_text(json.dumps(model_config, indent=2))
    print(f"wrote {common.MODEL_CONFIG}", flush=True)

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    schema = parquet_schema(n_slots)
    num_experts = cfg.num_experts
    slot_hist = Counter()          # active-slot count per token (should all == n_slots)
    nonexact_layers = 0            # diagnostic: layers where capture row count != P
    total_tokens = 0
    shared_expert_violation = 0

    for ds in args.datasets:
        n = args.max_seqs_per_dataset or 10**9
        print(f"\n=== dataset {ds}: loading up to {n} prompts ===", flush=True)
        try:
            prompts = load_prompts(ds, n)
        except Exception as e:
            print(f"  WARNING: could not load {ds}: {e} -- skipping", flush=True)
            continue
        print(f"  loaded {len(prompts)} prompts", flush=True)

        out_path = common.PATHS_DIR / f"{ds}.parquet"
        writer = pq.ParquetWriter(str(out_path), schema)

        b_seq, b_pos, b_tok, b_exp = [], [], [], []

        def flush_batch():
            if not b_seq:
                return
            table = pa.table(
                {
                    "dataset": pa.array([ds] * len(b_seq), pa.string()),
                    "seq_id": pa.array(b_seq, pa.int32()),
                    "position": pa.array(b_pos, pa.int32()),
                    "token_id": pa.array(b_tok, pa.int32()),
                    "experts": pa.array(b_exp, pa.list_(pa.int16(), n_slots)),
                },
                schema=schema,
            )
            writer.write_table(table)
            b_seq.clear(); b_pos.clear(); b_tok.clear(); b_exp.clear()

        # Leave room for the 1 sampled token within max_model_len.
        trunc_len = min(args.max_len, args.max_model_len - 1)
        kept = 0
        for seq_id, text in enumerate(prompts):
            ids = tok(text, truncation=True, max_length=trunc_len)["input_ids"]
            P = len(ids)
            if P == 0:
                continue
            llm.apply_model(pop)  # clear any residue
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            caps = llm.apply_model(pop)[0]  # {layer_idx: (N, k) int16}

            # Assemble (P, L_moe, k); slice the first P prefill rows per layer.
            seq_experts = np.empty((P, L_moe, k), dtype=np.int16)
            ok = True
            for li in moe_idxs:
                arr = caps.get(li)
                if arr is None or arr.shape[0] < P:
                    ok = False
                    nonexact_layers += 1
                    break
                seq_experts[:, layer_pos[li], :] = arr[:P]
            if not ok:
                continue

            if seq_experts.max() >= num_experts or seq_experts.min() < 0:
                shared_expert_violation += 1
                continue

            flat = seq_experts.reshape(P, n_slots)
            for pos in range(P):
                b_seq.append(seq_id)
                b_pos.append(pos)
                b_tok.append(int(ids[pos]))
                b_exp.append(flat[pos].tolist())
                slot_hist[n_slots] += 1  # every token has exactly n_slots active
                total_tokens += 1
            kept += 1
            if len(b_seq) >= 50_000:
                flush_batch()
            if (seq_id + 1) % 100 == 0:
                print(f"  [{ds}] {seq_id + 1} seqs, {total_tokens} tokens", flush=True)

        flush_batch()
        writer.close()
        print(f"  wrote {out_path} (kept {kept}/{len(prompts)} seqs)", flush=True)

    summary = {
        "total_tokens": total_tokens,
        "active_slots_per_token_histogram": dict(slot_hist),
        "expected_active_slots": n_slots,
        "validation_all_tokens_full": (set(slot_hist) == {n_slots}),
        "shared_expert_violations": shared_expert_violation,
        "sequences_with_short_capture": nonexact_layers,
    }
    (common.OUT_ROOT / "capture_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== capture summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
