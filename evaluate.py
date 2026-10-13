"""Dataset evaluation: shared discrete-ID sender, isolated receiver, score and verify."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import link

ALIASES = {'cat': ['cat'], 'dog': ['dog']}
PET_QUESTION = 'Is the pet in the image a cat or a dog? Answer with only cat or dog.'
POPE_SUFFIX = ' Answer with only yes or no.'
CONTENT_PATTERN = (r'(?:the pet in the image is |the animal in the image is |'
                   r'the image shows |it is )?(?:a |an |the )?'
                   r'(?:(?:small|large|big|little|black|white|brown|gray|grey|orange|ginger|fluffy|striped|spotted) ){0,3}(cat|dog)[.!]?')


def utc():
    return datetime.now(timezone.utc).isoformat()


def strict_label(answer, aliases):
    normalized = answer.lower().strip().strip('\"\'').rstrip('.!').strip()
    normalized = re.sub(r'^(a|an|the)\s+', '', normalized)
    matches = [label for label, forms in aliases.items() if normalized in forms]
    return matches[0] if len(matches) == 1 else None


def content_label(answer):
    match = re.fullmatch(CONTENT_PATTERN, answer.lower().strip())
    return match.group(1) if match else None


def pope_label(answer):
    text = answer.lower().strip().strip('\"\'').strip()
    match = re.match(r'^(yes|no)\b', text)
    if not match:
        return None
    label = match.group(1)
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    if label == 'yes' and any(w in {'no', 'not', 'never', "isn't", "aren't", "don't", "doesn't", "can't"} for w in words):
        return None
    if label == 'no' and 'yes' in words:
        return None
    return label


def official_answer(answer):
    text = answer.split('.')[0].replace(',', '')
    return 'no' if any(word in text.split(' ') for word in ('No', 'not', 'no')) else 'yes'


def question(sample, manifest, dataset):
    return sample['question'] + POPE_SUFFIX if dataset == 'pope' else manifest['question']


def answer_key(sample, manifest, dataset):
    if dataset == 'pope':
        return hashlib.sha256((sample['image'] + '\n' + question(sample, manifest, dataset)).encode()).hexdigest()
    return sample['sample_id']


def load_answers(path):
    records = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['record_id'] in records:
                raise ValueError('Duplicate answer record ID')
            records[item['record_id']] = item['record']
    return records


def append_answer(path, key, record):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps({'record_id': key, 'record': record}, ensure_ascii=False, separators=(',', ':')) + '\n')


def paths(args):
    manifest = args.manifest or link.ROOT / f'data/{args.dataset}/encodings/showo-{args.resolution}/manifest.json'
    output = args.output or link.ROOT / f'reports/showo-{args.resolution}/{args.dataset}'
    return Path(manifest), Path(output)


def evaluation_manifest(manifest_path, output):
    manifest = link.read_json(manifest_path)
    if not (output / 'metrics.json').exists() and (output / 'protocol.json').exists():
        protocol = link.read_json(output / 'protocol.json')
        assert protocol['manifest_sha256'] == link.sha256_file(manifest_path)
        if protocol['limit']:
            manifest = {**manifest, 'samples': manifest['samples'][:protocol['limit']]}
    return manifest


def score_rows(manifest, records, dataset):
    rows = []
    for sample in manifest['samples']:
        key = answer_key(sample, manifest, dataset)
        value = records[key]
        expected_question = question(sample, manifest, dataset)
        if value['question'] != expected_question:
            raise ValueError('Saved answer question differs from manifest')
        prediction = (pope_label(value['answer']) if dataset == 'pope' else
                      content_label(value['answer']) if dataset == 'oxford-pets' else
                      strict_label(value['answer'], manifest['aliases']))
        row = {**sample, 'answer': value['answer'], 'prediction': prediction,
               'correct': prediction == sample['label'], 'truncated': value['truncated'], 'answer_key': key}
        if dataset == 'pope':
            row.update(official_prediction=official_answer(value['answer']),
                       official_correct=official_answer(value['answer']) == sample['label'])
        else:
            strict = strict_label(value['answer'], manifest['aliases'])
            row.update(strict_prediction=strict, strict_correct=strict == sample['label'])
        rows.append(row)
    return rows


def metrics(rows, dataset):
    if not rows:
        raise ValueError('Empty evaluation')
    if dataset == 'oxford-pets':
        return pet_metrics(rows)
    if dataset == 'pope':
        return pope_metrics(rows)
    labels = sorted({r['label'] for r in rows})
    return {'samples': len(rows), 'correct': sum(r['correct'] for r in rows),
            'accuracy': sum(r['correct'] for r in rows) / len(rows),
            'invalid_answers': sum(r['prediction'] is None for r in rows),
            'per_class': {label: {'correct': sum(r['correct'] and r['label'] == label for r in rows),
                                  'total': sum(r['label'] == label for r in rows)} for label in labels}}


def score(args):
    manifest_path, output = paths(args)
    manifest = evaluation_manifest(manifest_path, output)
    records = load_answers(output / 'answers.jsonl')
    rows = score_rows(manifest, records, args.dataset)
    result = {'overall': metrics(rows, args.dataset)}
    if args.dataset == 'pope':
        result['categories'] = {c: metrics([r for r in rows if r['category'] == c], args.dataset)
                                for c in ('random', 'popular', 'adversarial')}
    return result, rows


def writable(output):
    if (output / 'metrics.json').exists():
        raise FileExistsError('Published baseline is read-only; specify a new --output directory')


def freeze(args, manifest_path, output):
    writable(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = link.read_json(manifest_path)
    value = {'manifest_sha256': link.sha256_file(manifest_path), 'dataset': args.dataset,
             'resolution': args.resolution, 'limit': args.limit, 'max_new_tokens': manifest.get('max_new_tokens', 16),
             'source_sha256': {p: link.sha256_file(link.ROOT / p) for p in ('link.py', 'evaluate.py')},
             'model': link.showo_metadata(args.resolution), 'pet_content_pattern': CONTENT_PATTERN,
             'decoding': {'top_k': 1, 'temperature': 1.0, 'seed': 0}, 'frozen_at_utc': utc()}
    path = output / 'protocol.json'
    if path.exists():
        old = link.read_json(path)
        assert all(old[k] == value[k] for k in value if k != 'frozen_at_utc'), 'Run protocol changed'
    else:
        link.write_json(path, value)
    assert manifest['resolution'] == args.resolution
    if args.limit:
        manifest = {**manifest, 'samples': manifest['samples'][:args.limit]}
    return manifest


def encode(args):
    manifest_path, output = paths(args)
    manifest = freeze(args, manifest_path, output)
    torch = link.initialize()
    meter = link.Meter(torch)
    vq = link.load_vq(torch)
    seen, encoded, reused = set(), 0, 0
    for sample in manifest['samples']:
        if sample['ids'] in seen:
            continue
        seen.add(sample['ids'])
        image = link.ROOT / sample['image']
        assert link.sha256_file(image) == sample['image_sha256']
        path = link.ROOT / sample['ids']
        if path.exists():
            packet = link.read_json(path)
            reused += 1
        else:
            packet = link.encode_image(torch, vq, image, resolution=args.resolution)
            link.write_json(path, packet)
            encoded += 1
        link.validate_packet(packet, resolution=args.resolution)
        assert packet['source_sha256'] == sample['image_sha256']
    link.write_json(output / 'encoding.json', {'encoded': encoded, 'reused': reused, 'metrics': meter.finish()})


def answer(args):
    manifest_path, output = paths(args)
    manifest = freeze(args, manifest_path, output)
    link.receiver_audit()
    torch = link.initialize()
    meter = link.Meter(torch)
    model, prompt = link.load_showo(torch, resolution=args.resolution)
    path = output / 'answers.jsonl'
    records = load_answers(path) if path.exists() else {}
    protocol_sha = link.sha256_file(output / 'protocol.json')
    for i, sample in enumerate(manifest['samples']):
        key = answer_key(sample, manifest, args.dataset)
        packet_path = link.ROOT / sample['ids']
        packet = link.read_json(packet_path)
        link.validate_packet(packet, resolution=args.resolution)
        assert packet['source_sha256'] == sample['image_sha256']
        if key in records:
            assert records[key]['protocol_sha256'] == protocol_sha
            assert records[key]['packet_sha256'] == link.sha256_file(packet_path)
            assert records[key]['question'] == question(sample, manifest, args.dataset)
            continue
        value = link.answer_packet(torch, model, prompt, packet, question(sample, manifest, args.dataset),
                                   manifest.get('max_new_tokens', 16), resolution=args.resolution)
        value.update(image=sample['image'], ids=sample['ids'], packet_sha256=link.sha256_file(packet_path),
                     protocol_sha256=protocol_sha, receiver_image_access_guard=True,
                     receiver_imported_visual_encoder='models.modeling_magvitv2' in sys.modules)
        assert not value['receiver_imported_visual_encoder']
        append_answer(path, key, value)
        records[key] = value
        if (i + 1) % 50 == 0:
            print(f'answer: {i + 1}/{len(manifest["samples"])}', flush=True)
    rows = score_rows(manifest, records, args.dataset)
    link.write_json(output / 'results.json', {'summary': {'overall': metrics(rows, args.dataset)}, 'rows': rows})
    link.write_json(output / 'inference.json', {'metrics': meter.finish(), 'images': len({r['image'] for r in rows})})


def reference(args):
    manifest_path, output = paths(args)
    manifest = freeze(args, manifest_path, output)
    records = load_answers(output / 'answers.jsonl')
    chosen = manifest.get('reference_sample_ids', [])
    samples = [s for s in manifest['samples'] if s['sample_id'] in chosen] if chosen else manifest['samples'][:6]
    if not samples:
        samples = manifest['samples'][:6]
    torch = link.initialize()
    vq = link.load_vq(torch)
    model, prompt = link.load_showo(torch, resolution=args.resolution)
    comparisons = []
    for sample in samples:
        packet = link.encode_image(torch, vq, link.ROOT / sample['image'], resolution=args.resolution)
        assert packet == link.read_json(link.ROOT / sample['ids'])
        value = link.answer_packet(torch, model, prompt, packet, question(sample, manifest, args.dataset),
                                   manifest.get('max_new_tokens', 16), resolution=args.resolution)
        received = records[answer_key(sample, manifest, args.dataset)]
        same = all(value[k] == received[k] for k in value)
        comparisons.append({'sample_id': sample['sample_id'], 'passed': same})
    result = {'passed': bool(comparisons) and all(r['passed'] for r in comparisons), 'rows': comparisons}
    link.write_json(output / 'reference.json', result)
    assert result['passed']


def verify(args):
    manifest_path, output = paths(args)
    manifest = evaluation_manifest(manifest_path, output)
    assert manifest['resolution'] == args.resolution
    records = load_answers(output / 'answers.jsonl')
    stored = link.read_json(output / 'results.json')
    metadata = link.read_json(output / 'metrics.json') if (output / 'metrics.json').exists() else None
    if metadata:
        assert metadata['manifest_sha256'] == link.sha256_file(manifest_path)
    rescored = score_rows(manifest, records, args.dataset)
    image_hashes, packet_hashes = {}, {}
    for sample, old, new in zip(manifest['samples'], stored['rows'], rescored):
        assert old['sample_id'] == new['sample_id'] == sample['sample_id']
        for k in ('answer', 'prediction', 'correct', 'truncated'):
            assert old[k] == new[k], (sample['sample_id'], k)
        for k in ('official_prediction', 'official_correct', 'strict_prediction', 'strict_correct'):
            if k in old:
                assert old[k] == new[k], (sample['sample_id'], k)
        value = records[new['answer_key']]
        if sample['image'] not in image_hashes:
            image_hashes[sample['image']] = link.sha256_file(link.ROOT / sample['image'])
        assert image_hashes[sample['image']] == sample['image_sha256']
        packet = link.read_json(link.ROOT / sample['ids'])
        link.validate_packet(packet, resolution=args.resolution)
        assert packet['source_sha256'] == sample['image_sha256']
        assert value['input_ids'][2:2 + len(packet['ids'])] == [x + 50305 for x in packet['ids']]
        assert value['showo_revision'] == link.showo_metadata(args.resolution)['revision']
        assert value['decoding'] == {'top_k': 1, 'temperature': 1.0, 'seed': 0, 'max_new_tokens': manifest.get('max_new_tokens', 16)}
        if args.dataset != 'imagenette':
            assert value['receiver_image_access_guard'] and not value['receiver_imported_visual_encoder']
            if sample['ids'] not in packet_hashes:
                packet_hashes[sample['ids']] = link.sha256_file(link.ROOT / sample['ids'])
            assert value['packet_sha256'] == packet_hashes[sample['ids']]
            if metadata:
                assert value['protocol_sha256'] == metadata['original_protocol_sha256']
    assert len(stored['rows']) == len(rescored) == len(manifest['samples'])
    assert set(records) == {answer_key(s, manifest, args.dataset) for s in manifest['samples']}
    current = metrics(rescored, args.dataset)
    expected = stored['summary'].get('overall', stored['summary'])
    assert all(current[k] == expected[k] for k in current if k in expected)
    return {'passed': True, 'dataset': args.dataset, 'resolution': args.resolution,
            'samples': len(rescored), 'unique_answers': len(records), 'score': current}


def run(args):
    _, output = paths(args)
    writable(output)
    for stage in ('encode', 'answer'):
        cmd = [sys.executable, '-B', str(Path(__file__)), stage, '--dataset', args.dataset,
               '--resolution', str(args.resolution), '--output', str(output)]
        if args.manifest:
            cmd.extend(['--manifest', str(args.manifest)])
        if args.limit:
            cmd.extend(['--limit', str(args.limit)])
        subprocess.run(cmd, cwd=link.ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['encode', 'answer', 'reference', 'score', 'verify', 'run'])
    parser.add_argument('--dataset', choices=['oxford-pets', 'pope', 'imagenette'], required=True)
    parser.add_argument('--resolution', type=int, choices=[256, 512], default=link.DEFAULT_RESOLUTION)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='New-run diagnostic subset only; published baseline scoring always uses its full manifest')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    result = globals()[args.command](args)
    if args.command == 'score':
        result = result[0]
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


# Frozen metric implementations are copied below from the pre-cleanup sources.
def pet_metrics(rows):
    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    p, z = correct/n, 1.96
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    radius = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/denom
    per_class = {label: {'samples': sum(r['label']==label for r in rows),
                        'correct': sum(r['correct'] and r['label']==label for r in rows),
                        'invalid': sum(r['prediction'] is None and r['label']==label for r in rows)} for label in ALIASES}
    for c in per_class.values():
        c['accuracy'] = c['correct']/c['samples'] if c['samples'] else None
    return {'samples': n, 'correct': correct, 'accuracy': p,
            'balanced_accuracy': (sum(c['accuracy'] for c in per_class.values())/2
                                  if all(c['accuracy'] is not None for c in per_class.values()) else None),
            'per_species': per_class, 'invalid': sum(r['prediction'] is None for r in rows),
            'invalid_rate': sum(r['prediction'] is None for r in rows)/n,
            'strict_correct': sum(r['strict_correct'] for r in rows),
            'strict_accuracy': sum(r['strict_correct'] for r in rows)/n,
            'strict_invalid': sum(r['strict_prediction'] is None for r in rows),
            'truncated': sum(r['truncated'] for r in rows),
            'wilson_95_interval': [center-radius, center+radius],
            'confusion': {label: {pred: sum(r['label']==label and (r['prediction'] or 'invalid')==pred for r in rows)
                                 for pred in ('cat','dog','invalid')} for label in ALIASES}}

def interval(rows):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row['image']].append(row['correct'])
    values = [sum(v) / len(v) for v in by_image.values()]
    rng = random.Random(20261012)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return [draws[49], draws[1949]]

def pope_metrics(rows):
    tp = sum(r['prediction'] == 'yes' and r['label'] == 'yes' for r in rows)
    fp = sum(r['prediction'] == 'yes' and r['label'] == 'no' for r in rows)
    tn = sum(r['prediction'] == 'no' and r['label'] == 'no' for r in rows)
    fn = sum(r['prediction'] == 'no' and r['label'] == 'yes' for r in rows)
    invalid = sum(r['prediction'] is None for r in rows)
    positive = sum(r['label'] == 'yes' for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / positive if positive else 0
    return {'questions': len(rows), 'images': len({r['image'] for r in rows}),
            'correct': sum(r['correct'] for r in rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows),
            'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
            'yes_ratio': (tp+fp)/len(rows), 'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn,
            'invalid': invalid, 'invalid_positive': sum(r['prediction'] is None and r['label']=='yes' for r in rows),
            'invalid_negative': sum(r['prediction'] is None and r['label']=='no' for r in rows),
            'truncated': sum(r['truncated'] for r in rows),
            'official_accuracy': sum(r['official_correct'] for r in rows)/len(rows),
            'image_bootstrap_95_interval': interval(rows)}

if __name__ == '__main__':
    main()
