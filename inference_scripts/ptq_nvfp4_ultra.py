#!/usr/bin/env python
"""NVFP4 mixed-precision PTQ for Nemotron-3-Ultra (green-step42), matching the GA release recipe.

GA recipe (from NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/hf_quant_config.json, MIXED_PRECISION):
  - backbone.layers.*.mixer.experts.*.{up,down}_proj      -> NVFP4 (block 16)
  - backbone.layers.*.mixer.{in_proj,out_proj}            -> FP8 (per-tensor)
  - backbone.layers.*.mixer.shared_experts.{up,down}_proj -> FP8 (per-tensor)
  - attention qkv/o, router/gate, embeddings, lm_head, MTP -> BF16 (unquantized)
  - KV cache -> FP8

Modes:
  --validate-structure : load on META device (no weights/GPU), confirm the name patterns select
                         exactly the GA-recipe module set (counts). Cheap sanity gate.
  (default)            : full PTQ -- load BF16 weights, calibrate (static amax) on a small text
                         set, export a unified-HF NVFP4 checkpoint with hf_quant_config.json.
"""
import argparse
import copy
import fnmatch
import json
import os

import torch


# ---- module-name patterns (glob, modelopt-style) ----------------------------------------
NVFP4_PATTERNS = [
    "*mixer.experts.*.up_proj",
    "*mixer.experts.*.down_proj",
]
FP8_PATTERNS = [
    "*mixer.in_proj",
    "*mixer.out_proj",
    "*mixer.shared_experts.up_proj",
    "*mixer.shared_experts.down_proj",
]


def build_quant_cfg(mtq):
    """Compose the MIXED_PRECISION quant_cfg matching the GA recipe, CALIBRATION-FREE.

    - routed experts up/down -> NVFP4 (weight + DYNAMIC block-scaled activation) == GA exactly.
    - in/out_proj + shared_experts -> FP8 WEIGHT-ONLY (activation left BF16). GA uses static-FP8
      activations on these 192 non-expert projections; weight-only is a tiny, documented deviation
      that removes the need for a 550B calibration forward pass (kept tractable on one node).
    - everything else (attn, router/gate, latent proj, embeddings, lm_head, MTP) -> BF16.
    - KV-cache FP8 is applied at SERVE time (vLLM --kv-cache-dtype fp8), not baked here.
    algorithm "max" => compute WEIGHT amax from the weights themselves (no data/forward needed);
    activation quantizers are all dynamic (NVFP4) or disabled (FP8 weight-only) so no calibration
    forward is required. (algorithm=None skips weight amax too -> export AttributeError on _amax.)
    """
    nvfp4 = mtq.NVFP4_DEFAULT_CFG["quant_cfg"]
    fp8 = mtq.FP8_DEFAULT_CFG["quant_cfg"]
    nv_w, nv_i = nvfp4["*weight_quantizer"], nvfp4["*input_quantizer"]  # NVFP4 act is dynamic
    f8_w = fp8["*weight_quantizer"]

    quant_cfg = {
        "default": {"enable": False},
        "*weight_quantizer": {"enable": False},
        "*input_quantizer": {"enable": False},
    }
    for p in NVFP4_PATTERNS:
        quant_cfg[f"{p}.weight_quantizer"] = copy.deepcopy(nv_w)
        quant_cfg[f"{p}.input_quantizer"] = copy.deepcopy(nv_i)
    for p in FP8_PATTERNS:
        quant_cfg[f"{p}.weight_quantizer"] = copy.deepcopy(f8_w)
        quant_cfg[f"{p}.input_quantizer"] = {"enable": False}  # weight-only -> no calibration

    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def _linear_module_names(model):
    out = []
    for name, mod in model.named_modules():
        # treat anything with a 2D weight as a linear-like target
        w = getattr(mod, "weight", None)
        if w is not None and getattr(w, "ndim", 0) == 2:
            out.append(name)
    return out


def validate_structure(model_path):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("arch:", cfg.architectures, "| dtype:", getattr(cfg, "dtype", None))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    names = _linear_module_names(model)
    print("total linear-like modules:", len(names))

    def count(patterns):
        sel = set()
        for n in names:
            if any(fnmatch.fnmatch(n, p) for p in patterns):
                sel.add(n)
        return sel

    nv = count(NVFP4_PATTERNS)
    f8 = count(FP8_PATTERNS)
    print(f"NVFP4-selected (experts up/down): {len(nv)}  (GA recipe = 49152)")
    print(f"FP8-selected   (in/out/shared)  : {len(f8)}  (GA recipe = 192)")
    overlap = nv & f8
    print("overlap (should be 0):", len(overlap))
    rest = [n for n in names if n not in nv and n not in f8]
    print("unquantized linear-like (BF16) count:", len(rest))
    print("sample NVFP4:", sorted(nv)[:2])
    print("sample FP8  :", sorted(f8)[:6])
    print("sample BF16 :", rest[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir")
    ap.add_argument("--validate-structure", action="store_true")
    ap.add_argument("--calib-file", help="jsonl with a 'text' field; pre-staged (GPU nodes may lack internet)")
    ap.add_argument("--calib-size", type=int, default=512)
    ap.add_argument("--calib-seqlen", type=int, default=4096)
    args = ap.parse_args()

    if args.validate_structure:
        validate_structure(args.model_path)
        return

    # ---- full PTQ path (calibration-free) ----
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert args.output_dir, "--output-dir required for PTQ"
    # export runs a forward pass; cuDNN fused attention has no valid engine for this NemotronH
    # config on GB200/cu130 -> force the math SDPA backend (pure matmul+softmax).
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    config = build_quant_cfg(mtq)
    print("quant_cfg keys:", len(config["quant_cfg"]), "| algorithm:", config["algorithm"], flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("loading model (device_map=auto, bf16, eager attn, CPU offload as needed)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model.eval()

    # calibration-free: NVFP4 weight + dynamic activation, FP8 weight-only -> no forward needed
    print("quantizing (no calibration)...", flush=True)
    mtq.quantize(model, config, forward_loop=None)
    try:
        mtq.print_quant_summary(model)
    except Exception as e:
        print("print_quant_summary skipped:", e)

    os.makedirs(args.output_dir, exist_ok=True)
    print("exporting unified-HF NVFP4 ->", args.output_dir, flush=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
