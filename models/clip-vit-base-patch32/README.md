# OpenAI CLIP ViT-B/32 for the TiViT adapter

Files: `config.json`, `preprocessor_config.json`, `model.safetensors`.
`model.safetensors` (plural) is the Transformers filename. The checkpoint includes
both text and vision tensors; the TiViT adapter loads only the vision tower.
These are pretrained CLIP weights, not a trained precipitation classifier.

Source: https://huggingface.co/openai/clip-vit-base-patch32
Pinned revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
OpenAI does not supply safetensors at this revision. This file is a local,
lossless conversion of the official `pytorch_model.bin`, not an official
OpenAI safetensors release. Source SHA256, converted SHA256 and conversion
procedure are recorded in `provenance.json`. The OpenAI MIT license is included.

## Clone and use

Install Git LFS before cloning, or run after cloning:

```sh
git lfs install
git lfs pull
python -m pip install -r requirements-tivit.txt
```

The precipitation notebook uses this local directory. The adapter explicitly
requires safetensors and loads local paths with `local_files_only=True`.
To reproduce the download and conversion, run
`python scripts/download_clip_checkpoint.py` (network access required).

## Security review and limitations

- Source: official OpenAI namespace over HTTPS; pinned revision. The original
  binary SHA256 was checked against Hugging Face's LFS metadata before loading.
  Hashes prove integrity relative to that source, not absence of malicious data.
- Conversion used PyTorch 2.14.1 with `weights_only=True` on CPU; no unrestricted
  pickle fallback. All 400 tensor keys and values were checked for exact equality
  after the safetensors roundtrip. The original pickle file stays in an ignored cache.
- Runtime loads only JSON and safetensors with built-in Transformers CLIP classes.
  No downloaded Python code or `trust_remote_code=True` is needed. Safetensors
  avoids pickle code execution; it does not eliminate library vulnerabilities or
  resource-exhaustion risks. Keep PyTorch, Transformers and Safetensors patched.
- Offline model loading and a finite vision forward pass were verified. This is a
  targeted artifact/loading review, not a full dependency vulnerability audit,
  penetration test, antivirus certificate or proof that weights lack backdoors.
- CLIP was trained on image/text data and is not validated for weather forecasting.
  Rainfall performance, rare-class recall and probability calibration require our
  chronological task-specific evaluation. Do not treat it as a weather warning model.
- Git LFS stores the large binary separately. Cloning requires LFS; account storage
  and download quotas still apply. No API keys or credentials are in these artifacts.

References:
- https://huggingface.co/docs/safetensors/index
- https://docs.pytorch.org/docs/stable/generated/torch.load.html
- https://huggingface.co/openai/clip-vit-base-patch32
- https://github.com/openai/CLIP/blob/main/model-card.md
