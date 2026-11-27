"""Time the hidden-state pass a BELT trainer would make on Qwen3.6-35B-A3B.

BELT's trainer keeps the backbone frozen and, every epoch, runs it over each
training chain to read one layer's states. Only the layers up to that one are
needed, so the model is loaded truncated to ``--layers`` decoder layers. This
measures load memory and tokens per second for full-sequence forwards at
several lengths, in the trainer's no-grad setting.

    python scripts/belt_tutorial/qwen36_hf_reader_spike.py --layers 24 --out hf.json
"""

import argparse
import json
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--lengths", default="2048,8192,16384")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    text = getattr(config, "text_config", config)
    text.num_hidden_layers = args.layers
    if getattr(text, "layer_types", None):
        text.layer_types = text.layer_types[: args.layers]
    report = {
        "model": MODEL,
        "layers": args.layers,
        "gpu": torch.cuda.get_device_name(0),
    }

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, config=config, device_map="cuda:0", dtype="auto"
    ).eval()
    report["load_seconds"] = time.perf_counter() - t0
    report["model_class"] = type(model).__name__
    report["weights_gb"] = torch.cuda.memory_allocated() / 2**30
    body = model.get_decoder() if hasattr(model, "get_decoder") else model.model

    rows = []
    for length in [int(x) for x in args.lengths.split(",")]:
        ids = torch.randint(1000, 50000, (1, length), device="cuda:0")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            body(input_ids=ids, use_cache=False)  # warm-up
            torch.cuda.synchronize()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = body(input_ids=ids, use_cache=False)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        assert out.last_hidden_state.shape[-1] == text.hidden_size
        best = min(times)
        rows.append(
            {
                "tokens": length,
                "seconds": best,
                "tokens_per_second": length / best,
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    report["forward"] = rows
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
