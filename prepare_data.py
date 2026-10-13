"""Prepare model manifests from local official dataset resources, or verify them."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import tarfile
import zipfile

import link


def prepare(args):
    root = link.ROOT / 'data' / args.dataset
    path = root / f'encodings/showo-{args.resolution}/manifest.json'
    if path.exists():
        raise FileExistsError(f'Manifest already exists: {path}; use verify')
    manifest = link.read_json(root / 'manifest.json')
    if args.dataset == 'oxford-pets':
        missing = {Path(s['image']).name for s in manifest['samples'] if not (link.ROOT / s['image']).exists()}
        if missing:
            with tarfile.open(root / 'archives/images.tar.gz', 'r:gz') as archive:
                for member in archive:
                    name = Path(member.name).name
                    if name in missing and member.isfile() and member.name == 'images/' + name:
                        destination = root / 'images' / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(archive.extractfile(member).read())
    samples = []
    for sample in manifest['samples']:
        image_path = link.ROOT / sample['image']
        if not image_path.exists():
            raise FileNotFoundError(f'Missing image: {image_path}; obtain the official resource before preparation')
        sha = link.sha256_file(image_path)
        if sample.get('image_sha256') and sample['image_sha256'] != sha:
            raise ValueError(f'Image checksum mismatch: {image_path}')
        samples.append({**sample, 'image_sha256': sha,
                        'ids': f'data/{args.dataset}/encodings/showo-{args.resolution}/ids/{image_path.stem}.json'})
    result = {**manifest, 'resolution': args.resolution,
              'num_visual_ids': link.visual_spec(args.resolution)['num_ids'], 'samples': samples}
    link.write_json(path, result)
    return {'manifest': str(path), 'samples': len(samples)}


def verify(args):
    path = link.ROOT / f'data/{args.dataset}/encodings/showo-{args.resolution}/manifest.json'
    manifest = link.read_json(path)
    assert manifest['resolution'] == args.resolution
    assert len({s['sample_id'] for s in manifest['samples']}) == len(manifest['samples'])
    seen = set()
    for sample in manifest['samples']:
        if sample['image'] in seen:
            continue
        seen.add(sample['image'])
        assert link.sha256_file(link.ROOT / sample['image']) == sample['image_sha256']
        packet_path = link.ROOT / sample['ids']
        if packet_path.exists():
            packet = link.read_json(packet_path)
            link.validate_packet(packet, resolution=args.resolution)
            assert packet['source_sha256'] == sample['image_sha256']
        elif args.require_ids:
            raise FileNotFoundError(packet_path)
    return {'passed': True, 'dataset': args.dataset, 'resolution': args.resolution,
            'images': len(seen), 'samples': len(manifest['samples']),
            'labels': dict(Counter(s['label'] for s in manifest['samples']))}


def verify_archive():
    index = link.read_json(link.ROOT / 'archive/manifest.json')
    path = link.ROOT / index['archive']
    assert link.sha256_file(path) == index['archive_sha256']
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == set(index['files'])
        for name, item in index['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item['sha256'], name
    return {'passed': True, 'files': len(index['files']), 'archive_sha256': index['archive_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'archive-verify'])
    parser.add_argument('--dataset', choices=['oxford-pets', 'pope', 'imagenette'])
    parser.add_argument('--resolution', type=int, choices=[256, 512], default=link.DEFAULT_RESOLUTION)
    parser.add_argument('--require-ids', action='store_true')
    args = parser.parse_args()
    if args.command != 'archive-verify' and not args.dataset:
        parser.error('--dataset is required for prepare/verify')
    import json
    result = verify_archive() if args.command == 'archive-verify' else globals()[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
