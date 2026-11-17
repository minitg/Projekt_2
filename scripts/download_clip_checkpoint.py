"""Download a pinned official OpenAI CLIP snapshot and convert to safetensors."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models' / 'clip-vit-base-patch32'
SOURCE = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    print(f'Downloading {path.name}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def main():
    import torch
    from safetensors.torch import save_file, load_file
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{SOURCE}/revision/{REVISION}?blobs=true', timeout=60) as response:
        info = json.load(response)
    revision = info['sha']
    expected = next(f['lfs']['sha256'] for f in info['siblings'] if f['rfilename'] == 'pytorch_model.bin')
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'preprocessor_config.json'):
        download(f'https://huggingface.co/{SOURCE}/resolve/{revision}/{name}', DEST / name)
        json.loads((DEST / name).read_text())
    original = ROOT / '.cache' / 'foundation_models' / 'clip-original.bin'
    original.parent.mkdir(parents=True, exist_ok=True)
    if not original.exists() or sha(original) != expected:
        download(f'https://huggingface.co/{SOURCE}/resolve/{revision}/pytorch_model.bin', original)
    if sha(original) != expected:
        raise RuntimeError('Original checkpoint SHA256 mismatch')
    # Never fall back to unrestricted pickle loading.
    state = torch.load(original, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a plain tensor state dictionary')
    target = DEST / 'model.safetensors'
    save_file({k: v.contiguous() for k, v in state.items()}, target, metadata={'format': 'pt'})
    restored = load_file(target)
    assert set(state) == set(restored)
    assert all(torch.equal(state[k], restored[k]) for k in state)
    download('https://raw.githubusercontent.com/openai/CLIP/main/LICENSE', DEST / 'LICENSE')
    provenance = {'source': SOURCE, 'revision': revision, 'original_sha256': expected,
                  'conversion': 'torch.load(weights_only=True, map_location=cpu); safetensors.save_file; exact tensor roundtrip',
                  'torch_version': torch.__version__, 'tensor_count': len(state),
                  'files_sha256': {n: sha(DEST / n) for n in ('config.json', 'preprocessor_config.json', 'model.safetensors', 'LICENSE')}}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
