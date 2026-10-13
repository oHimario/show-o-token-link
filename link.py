"""Original discrete Show-o: image -> raw IDs -> answer, using pinned local weights.

The answer entry point never imports MAGVIT or opens an image. Official prompting,
attention mask, model classes and top_k=1 decoding are reused without modification.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream' / 'Show-o'
DEFAULT_RESOLUTION = 512
os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'show-o-token-link-hf'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def namespace(name, path):
    # Skip upstream models/__init__.py, which imports both visual encoders.
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def initialize():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this hardware feasibility run.')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    namespace('models', UPSTREAM / 'models')
    namespace('training', UPSTREAM / 'training')
    return torch


class Meter:
    def __init__(self, torch):
        self.torch = torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.stop = threading.Event()
        self.device_peak_mib = None
        self.cpu_peak_mib = 0
        self.monitor = threading.Thread(target=self.sample, daemon=True)
        self.monitor.start()

    def sample(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            self.cpu_peak_mib = max(self.cpu_peak_mib, process.memory_info().rss / 2**20)
            try:
                text = subprocess.check_output(
                    ['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    text=True, timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                used = float(text.strip())
                self.device_peak_mib = max(self.device_peak_mib or 0, used)
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.2)

    def finish(self):
        self.torch.cuda.synchronize()
        self.stop.set()
        self.monitor.join(timeout=4)
        return {
            'seconds': round(time.perf_counter() - self.start, 4),
            'peak_allocated_mib': round(self.torch.cuda.max_memory_allocated() / 2**20, 2),
            'peak_reserved_mib': round(self.torch.cuda.max_memory_reserved() / 2**20, 2),
            'sampled_device_peak_mib': self.device_peak_mib,
            'sampled_cpu_rss_peak_mib': round(self.cpu_peak_mib, 2),
            'measurement': 'PyTorch process allocator, includes model loading; excludes driver/other processes',
            'device_measurement': 'nvidia-smi device 0 total used memory sampled about every 0.2s plus query time; may miss brief peaks and includes other processes',
        }


def load_vq(torch):
    cls = importlib.import_module('models.modeling_magvitv2').MAGVITv2
    # Match the official FP32 default; avoid extra rounding before sign quantization.
    model, info = cls.from_pretrained(str(ROOT / 'weights' / 'magvitv2'),
                                     torch_dtype=torch.float32, output_loading_info=True)
    validate_loading(info)
    return model.eval().requires_grad_(False).to('cuda')


def validate_loading(info):
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint loading did not match: {info}')


def visual_spec(resolution):
    if resolution not in (256, 512):
        raise ValueError('Only official 256 and 512 discrete configurations are supported.')
    side = resolution // 16
    return {'resolution': resolution, 'grid': [side, side], 'num_ids': side * side,
            'checkpoint': 'showo' if resolution == 256 else 'showo-512'}


def showo_metadata(resolution):
    visual_spec(resolution)
    if resolution == 256:
        return read_json(ROOT / 'weights/manifest.json')['showo']
    info = read_json(ROOT / 'weights/showo-512/manifest.json')
    if not info.get('ready') or info['repo_id'] != 'showlab/show-o-512x512':
        raise RuntimeError('Pinned 512 checkpoint has not passed download verification.')
    return info


def encode_image(torch, model, image_path, resolution=DEFAULT_RESOLUTION):
    spec = visual_spec(resolution)
    from PIL import Image
    transform = importlib.import_module('training.utils').image_transform
    with Image.open(image_path) as original:
        pixels = transform(original.convert('RGB'), resolution=resolution).unsqueeze(0).to('cuda')
    with torch.inference_mode():
        ids = model.get_code(pixels).cpu().flatten().tolist()
    if len(ids) != spec['num_ids'] or any(not 0 <= code < 8192 for code in ids):
        raise RuntimeError('Unexpected visual code shape or range.')
    manifest = read_json(ROOT / 'weights' / 'manifest.json')
    return {
        'schema': 'showo-raw-ids-v1', 'ids': ids,
        'grid': spec['grid'], 'flatten_order': 'row-major', 'codebook_size': 8192,
        'preprocessing': {'resolution': resolution, 'resize': 'short-side bicubic',
                          'crop': f'center {resolution}x{resolution}', 'color': 'RGB', 'normalize': '[-1,1]'},
        'encoder_dtype': 'float32', 'source_sha256': sha256_file(image_path),
        'visual_tokenizer': {key: manifest['magvitv2'][key] for key in ('repo_id', 'revision')},
    }


def load_showo(torch, dtype='float16', resolution=DEFAULT_RESOLUTION):
    spec = visual_spec(resolution)
    showo_metadata(resolution)
    from transformers import AutoTokenizer
    prompting = importlib.import_module('training.prompting_utils')
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'weights' / 'phi'), padding_side='left')
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128)
    cls = importlib.import_module('models.modeling_showo').Showo
    checkpoint = ROOT / 'weights' / spec['checkpoint']
    config = read_json(checkpoint / 'config.json')
    if config['w_clip_vit'] or config['codebook_size'] != 8192 or config['num_vq_tokens'] != spec['num_ids']:
        raise RuntimeError(f'Checkpoint does not match the selected {resolution}px discrete Show-o.')
    model, info = cls.from_pretrained(str(checkpoint), llm_model_path=str(ROOT / 'weights' / 'phi'),
                                     torch_dtype=getattr(torch, dtype), output_loading_info=True)
    validate_loading(info)
    if hasattr(model, 'mm_projector'):
        raise RuntimeError('Continuous visual projector unexpectedly present.')
    if len(tokenizer) != config['llm_vocab_size'] + 10:
        raise RuntimeError('Tokenizer length does not match the checkpoint visual offset.')
    return model.eval().requires_grad_(False).to('cuda'), prompt


def validate_packet(packet, resolution=DEFAULT_RESOLUTION):
    spec = visual_spec(resolution)
    manifest = read_json(ROOT / 'weights' / 'manifest.json')
    expected = {key: manifest['magvitv2'][key] for key in ('repo_id', 'revision')}
    if packet['schema'] != 'showo-raw-ids-v1' or packet['visual_tokenizer'] != expected:
        raise ValueError('ID format or visual tokenizer revision mismatch.')
    ids = packet['ids']
    if (packet['grid'] != spec['grid'] or packet['codebook_size'] != 8192 or
            packet['preprocessing']['resolution'] != resolution or
            len(ids) != spec['num_ids'] or any(type(x) is not int or not 0 <= x < 8192 for x in ids)):
        raise ValueError('Invalid raw visual IDs.')


def answer_packet(torch, model, prompt, packet, question, max_new_tokens, resolution=DEFAULT_RESOLUTION):
    validate_packet(packet, resolution=resolution)
    if model.config.num_vq_tokens != visual_spec(resolution)['num_ids']:
        raise ValueError('Visual ID packet and receiving Show-o configuration differ.')
    tokenizer = prompt.text_tokenizer
    visual = torch.tensor([packet['ids']], device='cuda', dtype=torch.long) + len(tokenizer)
    text = tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids']
    spt = prompt.sptids_dict
    # Exact discrete branch formatting of official inference_mmu.py.
    idx = torch.cat([spt['<|mmu|>'].reshape(1, 1).to('cuda'),
                     spt['<|soi|>'].reshape(1, 1).to('cuda'), visual,
                     spt['<|eoi|>'].reshape(1, 1).to('cuda'),
                     spt['<|sot|>'].reshape(1, 1).to('cuda'),
                     torch.tensor(text, device='cuda', dtype=torch.long)], dim=1)
    if idx.shape[1] + max_new_tokens > model.showo.config.max_position_embeddings:
        raise ValueError('Question exceeds the model context window.')
    mask = importlib.import_module('training.prompting_utils').create_attention_mask_for_mmu(
        idx, eoi_id=spt['<|eoi|>'].item())
    # Reset after model loading: even top_k=1 may retain exact logit ties.
    torch.manual_seed(0)
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                               enabled=model_dtype == torch.float16):
        generated = model.mmu_generate(idx=idx, attention_mask=mask, max_new_tokens=max_new_tokens,
                                       temperature=1.0, top_k=1, eot_token=spt['<|eot|>'].item())
    tokens = [x.item() for x in generated]
    manifest = read_json(ROOT / 'weights' / 'manifest.json')
    return {
        'question': question, 'answer': tokenizer.decode(tokens, skip_special_tokens=True),
        'input_ids': idx.cpu().flatten().tolist(), 'generated_ids': tokens,
        'visual_vocab_offset': len(tokenizer),
        'decoding': {'top_k': 1, 'temperature': 1.0, 'seed': 0, 'max_new_tokens': max_new_tokens},
        'showo_dtype': str(model_dtype).replace('torch.', ''),
        'cuda_autocast': 'float16' if model_dtype == torch.float16 else None,
        'showo_revision': showo_metadata(resolution)['revision'],
        'phi_revision': manifest['phi']['revision'],
        'attention_implementation': model.showo.config._attn_implementation,
        'truncated': len(tokens) == max_new_tokens and tokens[-1] != spt['<|eot|>'].item(),
    }


def receiver_audit():
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            if Path(os.fsdecode(args[0])).suffix.lower() in extensions:
                raise RuntimeError('Receiver attempted to access an image.')
    sys.addaudithook(audit)


def environment():
    import platform
    import psutil
    import torch
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': {p.metadata['Name']: p.version for p in importlib.metadata.distributions()},
            'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
            'ram_total_gib': psutil.virtual_memory().total / 2**30,
            'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
            'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True, errors='replace'),
            'weights': read_json(ROOT / 'weights' / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    enc = sub.add_parser('encode')
    enc.add_argument('--image', required=True)
    enc.add_argument('--output', required=True)
    enc.add_argument('--resolution', type=int, choices=[256, 512], default=DEFAULT_RESOLUTION,
                     help='Model/input resolution (default: %(default)s)')
    for name in ('answer', 'reference'):
        p = sub.add_parser(name)
        p.add_argument('--ids' if name == 'answer' else '--image', required=True)
        p.add_argument('--question', required=True)
        p.add_argument('--max-new-tokens', type=int, default=32)
        p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
        p.add_argument('--resolution', type=int, choices=[256, 512], default=DEFAULT_RESOLUTION,
                       help='Model/input resolution (default: %(default)s)')
        p.add_argument('--output', required=True)
    comp = sub.add_parser('compare')
    comp.add_argument('--reference', required=True)
    comp.add_argument('--received', required=True)
    comp.add_argument('--output', required=True)
    env = sub.add_parser('environment')
    env.add_argument('--output', default='reports/project/environment.json')
    args = parser.parse_args()
    if args.command == 'compare':
        reference, received = read_json(args.reference), read_json(args.received)
        checks = {key: reference[key] == received[key] for key in
                  ('input_ids', 'generated_ids', 'answer', 'decoding', 'showo_revision', 'phi_revision',
                   'showo_dtype', 'cuda_autocast', 'attention_implementation')}
        value = {'passed': all(checks.values()), 'checks': checks,
                 'receiver_image_access_guard': received.get('receiver_image_access_guard', False),
                 'receiver_imported_visual_encoder': received.get('receiver_imported_visual_encoder', True)}
        value['passed'] &= value['receiver_image_access_guard'] and not value['receiver_imported_visual_encoder']
        write_json(args.output, value)
        print(json.dumps(value, indent=2))
        if not value['passed']:
            raise SystemExit(1)
        return
    if args.command == 'environment':
        write_json(args.output, environment())
        return
    if args.command == 'answer':
        receiver_audit()
    torch = initialize()
    meter = Meter(torch)
    if args.command == 'encode':
        vq = load_vq(torch)
        value = encode_image(torch, vq, args.image, resolution=args.resolution)
    else:
        if args.command == 'reference':
            vq = load_vq(torch)
            packet = encode_image(torch, vq, args.image, resolution=args.resolution)
            del vq
            torch.cuda.empty_cache()
        else:
            packet = read_json(args.ids)
        model, prompt = load_showo(torch, dtype=args.dtype, resolution=args.resolution)
        value = answer_packet(torch, model, prompt, packet, args.question, args.max_new_tokens,
                              resolution=args.resolution)
        value['receiver_image_access_guard'] = args.command == 'answer'
        value['receiver_imported_visual_encoder'] = 'models.modeling_magvitv2' in sys.modules
    value['metrics'] = meter.finish()
    write_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('ids', 'input_ids', 'generated_ids')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
