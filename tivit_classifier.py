"""Local TiViT adaptation: channel-wise images, frozen CLIP, logistic head.

Method/reference: https://github.com/ExplainableML/TiViT (MIT).
This compact HF CLIP variant is not a reproduction of the ViT-H benchmark.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from benchmark_timing import clock
from classification_data import load_classification_bundle, file_sha256
from evaluate_precipitation_baselines import probability_metrics


def time_images(x, segment=7, stride=1, size=224):
    """One grayscale image per channel; only the supplied past window is used."""
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim != 3 or not torch.isfinite(x).all() or not 1 <= segment <= x.shape[1] or stride < 1:
        raise ValueError('expected finite (cases, time, channels) and valid segment/stride')
    x = x.transpose(1, 2).reshape(-1, x.shape[1])
    median = x.median(dim=1, keepdim=True).values
    quartiles = torch.quantile(x, torch.tensor([.25, .75], device=x.device), dim=1)
    x = (x - median) / (quartiles[1] - quartiles[0]).unsqueeze(1).add(1e-5)
    padding = (-(x.shape[1] - segment)) % stride
    x = F.pad(x.unsqueeze(1), (padding, 0), mode='replicate').squeeze(1)
    image = x.unfold(1, segment, stride)
    lo = image.amin(dim=(1, 2), keepdim=True)
    hi = image.amax(dim=(1, 2), keepdim=True)
    image = ((image - lo) / (hi - lo + 1e-5)).pow(.8)
    return F.interpolate(image.unsqueeze(1), size=(size, size), mode='nearest').repeat(1, 3, 1, 1)


class TiViTEncoder:
    def __init__(self, model_id=None, layer=6, device='cpu', revision=None):
        from transformers import CLIPImageProcessor, CLIPVisionModel
        if model_id is None:
            model_id = str(Path(__file__).resolve().parent / 'models/clip-vit-base-patch32')
        local = Path(model_id).is_dir()
        self.processor = CLIPImageProcessor.from_pretrained(model_id, revision=revision, local_files_only=local)
        self.model = CLIPVisionModel.from_pretrained(model_id, revision=revision,
                                                     local_files_only=local, use_safetensors=True).to(device).eval()
        self.model.requires_grad_(False)
        if not 1 <= layer <= self.model.config.num_hidden_layers:
            raise ValueError('layer outside vision transformer depth')
        self.layer, self.device = layer, device
        self.metadata = {'model_id': model_id, 'revision': getattr(self.model.config, '_commit_hash', revision),
                         'layer': layer, 'aggregation': 'mean including CLS; channel concatenation; L2 normalization',
                         'segment': 7, 'stride_samples': 1, 'frozen': True}

    @torch.inference_mode()
    def transform(self, x, batch_size=8):
        if batch_size < 1 or not len(x):
            raise ValueError('positive batch size and nonempty cases required')
        result = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            # Process channels separately to keep peak image/activation memory small.
            channels = []
            for channel in range(batch.shape[2]):
                images = time_images(batch[:, :, channel:channel+1], size=self.model.config.image_size)
                # Processor includes CLIP normalization, without a second rescaling.
                inputs = self.processor(images=list(images), do_rescale=False, return_tensors='pt')
                hidden = self.model(**{k: v.to(self.device) for k, v in inputs.items()},
                                    output_hidden_states=True).hidden_states[self.layer]
                channels.append(hidden.mean(1).cpu().numpy())
            embedding = np.concatenate(channels, axis=1)
            embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
            result.append(embedding)
            print(f'TiViT: {min(start + batch_size, len(x))}/{len(x)} cases', flush=True)
        return np.concatenate(result)


def run(data_dir, output_dir, encoder, horizon=3, batch_size=8, max_cases=None):
    """Fit only on train; report validation and retrospective evaluation separately."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    if max_cases is not None and max_cases < 3:
        raise ValueError('max_cases must be at least 3')
    data = load_classification_bundle(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, encode_seconds = {}, {}
    for split in ('train', 'validation', 'evaluation'):
        x, y, cases = data.arrays(horizon, split)
        if max_cases is not None and len(x) > max_cases:
            indices = np.linspace(0, len(x)-1, max_cases).round().astype(int)
            x, y, cases = x[indices], y[indices], cases.iloc[indices].reset_index(drop=True)
        start = clock()
        splits[split] = (encoder.transform(x, batch_size), y, cases)
        encode_seconds[split] = clock() - start
    x, y, _ = splits['train']
    if set(y) != {0, 1, 2}:
        raise ValueError('training subset must include all three classes; increase max_cases')
    start = clock()
    head = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(x, y)
    head_fit_seconds = clock() - start
    if head.n_iter_.max() >= head.max_iter:
        raise RuntimeError('classification head did not converge')
    np.savez_compressed(output / 'head.npz', coefficients=head.coef_, intercept=head.intercept_, classes=head.classes_)
    rows = []
    for split in ('validation', 'evaluation'):
        x, y, cases = splits[split]
        start = clock()
        p = head.predict_proba(x)
        predict_seconds = clock() - start
        prediction = cases.copy()
        prediction['target_class'] = y
        prediction[['p_class0', 'p_class1', 'p_class2']] = p
        prediction.to_parquet(output / f'{split}_predictions.parquet', index=False)
        matrix = confusion_matrix(y, p.argmax(1), labels=[0, 1, 2])
        np.savetxt(output / f'{split}_confusion.csv', matrix, delimiter=',', fmt='%d')
        # Inference per case = frozen-encoder embedding + head; excludes data loading and model load.
        rows.append({'model': 'TiViT-CLIP-B32', 'horizon_h': horizon, 'split': split, **probability_metrics(y, p),
                     'encode_seconds': encode_seconds[split],
                     'inference_ms_per_case': 1000 * (encode_seconds[split] + predict_seconds) / len(y),
                     'head_fit_seconds': head_fit_seconds, 'device': str(encoder.device),
                     'torch_threads': torch.get_num_threads(), 'batch_size': batch_size})
    scores = pd.DataFrame(rows)
    scores.to_csv(output / 'scores.csv', index=False)
    (output / 'run.json').write_text(json.dumps({
        'encoder': encoder.metadata, 'dataset_manifest_sha256': file_sha256(Path(data_dir) / 'manifest.json'),
        'code_sha256': file_sha256(Path(__file__)), 'features': data.features.columns.tolist(),
        'horizon_h': horizon, 'classes': [0, 1, 2], 'max_cases_per_split': max_cases,
        'timing_seconds': {'encode': encode_seconds, 'head_fit': head_fit_seconds, 'fine_tuning': 0.0},
        'fit': 'train only; C=1; natural class frequencies; no tuning on evaluation',
        'case_selection': 'all eligible cases' if max_cases is None else 'evenly spaced label-independent technical subset; not full benchmark',
        'evaluation': 'retrospective 2024; previously explored; not untouched test',
        'variant': 'TiViT adaptation with smaller CLIP B32; not paper benchmark reproduction',
        'pretraining_overlap': 'not independently excluded',
    }, indent=2) + '\n')
    return scores
