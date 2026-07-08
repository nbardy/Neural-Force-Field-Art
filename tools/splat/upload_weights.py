# tools/splat/upload_weights.py — publish the PACKED CLIP weights to the HF Hub
# so the prompt→splats page (src/splat_page.ts) can fetch them in the browser.
#
#   uv run --with huggingface_hub python tools/splat/upload_weights.py
#
# Why HF and not a GitHub Release: release assets send NO CORS header, so a
# browser fetch() from the Pages site is blocked; HF serves with CORS (same
# host the text model already loads from). Re-run this after regenerating
# weights_train.bin / plan_train.json (compile_plan.py --train) — they are a
# matched pair and must be uploaded together. Auth: `huggingface-cli login`
# (token saved to ~/.cache/huggingface) or HF_TOKEN env.
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Nbardy/nff-clip-splat-weights"
MODEL_DIR = Path("models/mobileclip_s0")
# Vision: our packed WGSL weights. Text: the stock ONNX text tower + tokenizer/
# config so transformers.js can load the text encoder from THIS repo too (fp16,
# 84 MB, verified lossless in-browser at graphOptimizationLevel:"basic"). One
# origin for everything the page fetches — no dependency on the Xenova repo.
FILES = [
    "weights_train.bin",
    "plan_train.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/text_model_fp16.onnx",
]

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
for f in FILES:
    p = MODEL_DIR / f
    print(f"uploading {f} ({p.stat().st_size/1e6:.1f} MB)…")
    api.upload_file(path_or_fileobj=str(p), path_in_repo=f, repo_id=REPO, repo_type="model")
print(f"done → https://huggingface.co/{REPO}/tree/main")
print(f"fetch base: https://huggingface.co/{REPO}/resolve/main/")
