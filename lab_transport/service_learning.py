"""CPU/GPU-sized ablations of the existing entity network, on factual data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dynamics_experiment import Episode
from .dynamics_model import _intent, _target
from .state_v2 import ObservableHistory, progress_adjusted
from .torch_structured_adapter import build_model, collate, encode_transition
from .torch_training import configure_deterministic_cuda, geometry_from_map, observation_from_dict
from .training_data import _path, validate_dataset
from .transition_schema import Assignment, Transition

SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEQUENCE_WEIGHT = 123, 5, 128, .001, .3
THRESHOLDS = tuple(round(i / 20, 2) for i in range(1, 20))


def contract_hashes():
    root = Path(__file__).parent
    names = ('service_learning.py', 'service_dynamics.py', 'torch_structured_adapter.py',
             'torch_training.py', 'training_data.py', 'state_v2.py', 'dynamics_model.py',
             'dynamics_revision.py', 'transition_schema.py', 'simulation.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def read_episodes(dataset: Path):
    """Read only validated factual observations; enrich each episode chronologically."""
    check = validate_dataset(dataset)
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    geometries = {layout: geometry_from_map(json.loads(_path(dataset, info['file']).read_text(encoding='utf-8')))
                  for layout, info in manifest['maps'].items()}
    episodes = []
    for entry in manifest['episodes']:
        observer = ObservableHistory()
        rows = []
        with _path(dataset, entry['file']).open(encoding='utf-8') as handle:
            for line in handle:
                data = json.loads(line)
                current = observer.observe(observation_from_dict(data['observation']))
                future = observer.observe(observation_from_dict(data['next_observation']))
                actions = tuple(Assignment(**a) for a in data['assignments'])
                rows.append(Transition(current, actions, future, data['next_boundary'], data['termination']))
        episodes.append(Episode(entry['split'], entry['layout'], entry['seed'], entry['regime'],
                                entry['behavior_policy'], entry['episode_status'], entry['completed_tasks'], rows))
    return manifest, geometries, episodes, check


def service_keys(row):
    """Match structured adapter service item order using current state/actions."""
    obs = progress_adjusted(row.observation)
    intents = _intent(obs, row.assignments)
    samples, stations = {s.id:s for s in obs.samples}, {s.id:s for s in obs.stations}
    positions = {s.id:s.position for s in obs.stations}
    next_samples = {s.id:s for s in row.next_observation.samples}
    result = []
    for robot in obs.robots:
        phase, sid = intents[robot.id]
        if sid is None:
            continue
        sample = samples[sid]
        station = stations[sample.pickup if phase == 'to_pickup' else sample.dropoff]
        if (robot.position == _target(robot, (phase, sid), samples, positions)
                and station.available and sid in next_samples):
            result.append((robot.id, sid, phase, station.id))
    return result


def sequence_target(rows, start, key, horizon=3):
    """Factual outcome label; a future outage censors the conditional window."""
    robot_id, sample_id, phase, station_id = key
    target_status = 'in_transit' if phase == 'to_pickup' else 'completed'
    for offset in range(horizon):
        index = start + offset
        if index >= len(rows):
            return None
        row = rows[index]
        robots = {r.id:r for r in row.observation.robots}
        stations = {s.id:s for s in row.observation.stations}
        actual = {s.id:s for s in row.next_observation.samples}
        if robot_id not in robots or station_id not in stations or sample_id not in actual:
            return None
        robot, station = robots[robot_id], stations[station_id]
        if not station.available or robot.position != station.position:
            return None
        if offset and (robot.task_id != sample_id or robot.phase != phase):
            return None
        if actual[sample_id].status == target_status:
            return 1
    return 0


def examples_for(episodes, geometries, split):
    examples = []
    windows = {'positive': 0, 'negative': 0, 'censored': 0}
    for ep in episodes:
        if ep.split != split:
            continue
        for index, row in enumerate(ep.rows):
            items = encode_transition(geometries[ep.layout], row.observation, row.assignments, row.next_observation)
            keys = service_keys(row)
            service = [item for item in items if item['kind'] == 'service']
            if len(keys) != len(service):
                raise ValueError('service items do not match factual service keys')
            for item, key in zip(service, keys):
                label = sequence_target(ep.rows, index, key)
                item['sequence_target'] = label
                windows['censored' if label is None else 'positive' if label else 'negative'] += 1
            examples.extend(items)
    if not examples:
        raise ValueError(f'no examples for {split}')
    return examples, windows


def sequence_collate(items):
    inputs, labels = collate(items)
    services = [item for item in items if item['kind'] == 'service']
    labels['sequence_mask'] = torch.tensor([item.get('sequence_target') is not None for item in services], dtype=torch.bool)
    labels['sequence_target'] = torch.tensor([float(item.get('sequence_target') or 0) for item in services])
    return inputs, labels


def loss_value(network, inputs, labels, positive_weight=1.0, sequence_weight=0.0):
    outputs = network(inputs)
    count = labels['motion'].numel() + labels['service'].numel()
    if not count:
        raise ValueError('empty batch')
    total = torch.zeros((), device=outputs['motion'].device)
    if labels['motion'].numel():
        total = total + F.cross_entropy(outputs['motion'], labels['motion'], reduction='sum')
    if labels['service'].numel():
        pos = torch.tensor(positive_weight, device=outputs['service'].device)
        total = total + F.binary_cross_entropy_with_logits(outputs['service'], labels['service'],
                                                          pos_weight=pos, reduction='sum')
        mask = labels['sequence_mask']
        if sequence_weight and bool(mask.any()):
            features = inputs['service'][mask]
            ctx = network.context(inputs['service_entities'][mask], inputs['service_entities_mask'][mask])
            logits = []
            for offset in range(3):
                hypothetical = features.clone()
                elapsed = features[:, 1] * 10 + offset
                expected = (features[:, 2] * 10).clamp_min(.1)
                hypothetical[:, 1] = elapsed.clamp_max(30) / 10
                hypothetical[:, 3] = (elapsed / expected).clamp_max(4) / 4
                if offset:
                    hypothetical[:, 6] = 0
                logits.append(network.service(torch.cat((hypothetical, ctx), dim=-1)).squeeze(-1))
            log_survival = F.logsigmoid(-torch.stack(logits, dim=1)).sum(dim=1)
            probability = (-torch.expm1(log_survival)).clamp(1e-6, 1 - 1e-6)
            total = total + sequence_weight * F.binary_cross_entropy(
                probability, labels['sequence_target'][mask], reduction='sum')
    return total / count, count


def _move(values, device):
    return {key: value.to(device) for key, value in values.items()}


def _epoch(network, loader, device, optimizer=None, positive_weight=1.0, sequence_weight=0.0):
    network.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = _move(inputs, device), _move(labels, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            value, n = loss_value(network, inputs, labels, positive_weight, sequence_weight)
            if not bool(torch.isfinite(value)):
                raise ValueError('non-finite loss')
            if optimizer is not None:
                value.backward()
                optimizer.step()
            total += value.item() * n
            count += n
    return total / count


def validation_probabilities(network, examples, device):
    network.eval()
    service = [item for item in examples if item['kind'] == 'service']
    loader = DataLoader(service, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sequence_collate)
    probabilities, targets = [], []
    with torch.inference_mode():
        for inputs, labels in loader:
            output = network(_move(inputs, device))
            probabilities.extend(torch.sigmoid(output['service']).cpu().tolist())
            targets.extend(labels['service'].tolist())
    return probabilities, targets


def calibrate(probabilities, labels):
    if not probabilities or len(probabilities) != len(labels) or sum(labels) == 0 or sum(labels) == len(labels):
        raise ValueError('validation needs positive and negative service examples')
    records = []
    for threshold in THRESHOLDS:
        tp = sum(p >= threshold and y == 1 for p, y in zip(probabilities, labels))
        fp = sum(p >= threshold and y == 0 for p, y in zip(probabilities, labels))
        fn = sum(p < threshold and y == 1 for p, y in zip(probabilities, labels))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        records.append({'threshold':threshold, 'f1':f1, 'tp':tp, 'fp':fp, 'fn':fn})
    selected = min(records, key=lambda r: (-r['f1'], abs(r['threshold']-.5), r['threshold']))
    return {'threshold':selected['threshold'], 'selected_validation':selected,
            'validation_examples':len(labels), 'validation_positives':int(sum(labels)), 'grid':records}


def fit_all(dataset: Path, output: Path, device_name='cpu'):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite training output: {output}')
    if device_name not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    if device_name == 'cuda':
        configure_deterministic_cuda()
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
    device = torch.device(device_name)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest, geometries, episodes, checked = read_episodes(dataset)
    train, train_windows = examples_for(episodes, geometries, 'train')
    validation, validation_windows = examples_for(episodes, geometries, 'validation')
    train_services = [x for x in train if x['kind'] == 'service']
    positives = sum(x['target'] for x in train_services)
    negatives = len(train_services) - positives
    if not positives or not negatives:
        raise ValueError('training needs positive and negative service outcomes')
    weight = min(10.0, negatives / positives)
    dataset_hash = hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    code_hashes = contract_hashes()
    output.mkdir(parents=True, exist_ok=True)
    results, networks = {}, {}
    for name, positive_weight, seq_weight in [('original', 1., 0.), ('weighted', weight, 0.),
                                               ('sequence', weight, SEQUENCE_WEIGHT)]:
        torch.manual_seed(SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(SEED)
        generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                                  collate_fn=sequence_collate, num_workers=0)
        val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sequence_collate)
        network = build_model().to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        best_loss, best_state, best_epoch = math.inf, None, None
        reports = []
        started = time.monotonic()
        for epoch in range(1, EPOCHS+1):
            train_loss = _epoch(network, train_loader, device, optimizer, positive_weight, seq_weight)
            val_loss = _epoch(network, val_loader, device)
            reports.append({'epoch':epoch, 'train_objective':train_loss, 'validation_unweighted_loss':val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
        assert best_state is not None
        network.load_state_dict(best_state)
        network.eval()
        calibration = calibrate(*validation_probabilities(network, validation, device)) if name != 'original' else None
        checkpoint = output / f'{name}.pt'
        with checkpoint.open('xb') as handle:
            torch.save({'model_state':best_state, 'variant':name, 'dataset_manifest_sha256':dataset_hash,
                        'contract_code_sha256':code_hashes,
                        'positive_weight':positive_weight, 'sequence_weight':seq_weight,
                        'calibration':calibration, 'seed':SEED, 'best_epoch':best_epoch}, handle)
        results[name] = {'best_epoch':best_epoch, 'validation_unweighted_loss':best_loss,
                         'positive_weight':positive_weight, 'sequence_weight':seq_weight,
                         'calibration':calibration, 'epochs':reports, 'wall_seconds':time.monotonic()-started,
                         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        networks[name] = network
        print(f'{name}: best_epoch={best_epoch} val={best_loss:.6f} '
              f'threshold={calibration["threshold"] if calibration else 0.5}', flush=True)
    report = {'dataset_manifest_sha256':dataset_hash, 'contract_code_sha256':code_hashes, 'data_check':checked,
              'device':str(device), 'torch_version':torch.__version__, 'seed':SEED, 'epochs':EPOCHS,
              'batch_size':BATCH_SIZE, 'learning_rate':LEARNING_RATE, 'train_examples':len(train),
              'training_service_positive':positives, 'training_service_negative':negatives,
              'train_sequence_windows':train_windows, 'validation_sequence_windows':validation_windows,
              'variants':results}
    (output/'training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report, networks, geometries, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        report, _, _, _ = fit_all(args.dataset, args.output, args.device)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'service training failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
