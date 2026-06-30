"""Load the SABER NLP corpus (MT-Bench, Alpaca, XSUM, GLUE SST-2/MNLI).

Each loader returns a list of plain-text prompts. Tokenization to vLLM happens
in ``capture_paths.py`` with the Qwen tokenizer; here we only produce text.
"""
from __future__ import annotations


def _alpaca(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        text = instr + ("\n\n" + inp if inp else "")
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


def _xsum(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("EdinburghNLP/xsum", split="train", streaming=True,
                      trust_remote_code=True)
    out = []
    for ex in ds:
        doc = (ex.get("document") or "").strip()
        if doc:
            out.append(doc)
        if len(out) >= n:
            break
    return out


def _sst2(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "sst2", split="train")
    out = []
    for ex in ds:
        s = (ex.get("sentence") or "").strip()
        if s:
            out.append(s)
        if len(out) >= n:
            break
    return out


def _mnli(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "mnli", split="train")
    out = []
    for ex in ds:
        prem = (ex.get("premise") or "").strip()
        hyp = (ex.get("hypothesis") or "").strip()
        text = (prem + " " + hyp).strip()
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


def _mtbench(n: int) -> list[str]:
    from datasets import load_dataset

    # Try a few known sources; MT-Bench is small (~80 questions, multi-turn).
    candidates = [
        ("HuggingFaceH4/mt_bench_prompts", None, "train"),
        ("lmsys/mt_bench_human_judgments", None, "human"),
    ]
    last_err = None
    for name, cfg, split in candidates:
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        except Exception as e:  # pragma: no cover - network/availability dependent
            last_err = e
            continue
        out: list[str] = []
        for ex in ds:
            if "prompt" in ex:  # mt_bench_prompts: prompt is a list of turns
                p = ex["prompt"]
                turns = p if isinstance(p, list) else [p]
                out.extend(str(t).strip() for t in turns if str(t).strip())
            elif "conversation_a" in ex:  # human_judgments
                for turn in ex["conversation_a"]:
                    if turn.get("role") == "user":
                        out.append(str(turn.get("content", "")).strip())
            if len(out) >= n:
                break
        out = [t for t in out if t][:n]
        if out:
            return out
    raise RuntimeError(f"could not load MT-Bench from any source: {last_err}")


_LOADERS = {
    "alpaca": _alpaca,
    "xsum": _xsum,
    "sst2": _sst2,
    "mnli": _mnli,
    "mtbench": _mtbench,
}


def load_prompts(dataset: str, n: int) -> list[str]:
    """Return up to ``n`` text prompts for ``dataset``."""
    if dataset not in _LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {list(_LOADERS)}")
    return _LOADERS[dataset](n)
