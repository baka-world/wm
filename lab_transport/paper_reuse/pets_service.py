"""MBRL-Lib GaussianMLP service duration + existing legal rule transitions.

The neural network models log effective service duration; it never emits robot
moves or task states. GaussianMixture survival yields a conditional completion
probability, sampled only for observed available stations. The existing rule
model enforces movement, availability and lifecycle constraints.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from mbrl.models import GaussianMLP, ModelTrainer, OneDTransitionRewardModel
from mbrl.util.replay_buffer import BootstrapIterator, ReplayBuffer, TransitionIterator

from lab_transport.dynamics_model import _intent
from lab_transport.paper_reuse.service_dataset import FEATURES, service_features
from lab_transport.state_v2 import ObservationV2
from lab_transport.torch_structured_adapter import ProgressAwareRule, make_observer

ENSEMBLE = 5
HIDDEN = 32
LAYERS = 2


def build_network():
    return GaussianMLP(in_size=len(FEATURES)+1, out_size=1, device='cpu',
                       num_layers=LAYERS, ensemble_size=ENSEMBLE, hid_size=HIDDEN,
                       deterministic=False, propagation_method=None)


def buffer_for(rows, seed):
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(len(rows),(1,),(len(FEATURES),),rng=rng)
    for row in rows:
        # OneDTransitionRewardModel concatenates obs and action. Here zero is
        # a placeholder; action slots encode observable service covariates.
        buffer.add(np.zeros(1,dtype=np.float32),np.asarray(row['features'],dtype=np.float32),
                   np.asarray([math.log(row['duration_active_ticks'])],dtype=np.float32),0.,False,False)
    return buffer


def train(dataset: Path, checkpoint: Path, *, seed=17, epochs=35):
    if checkpoint.exists():
        raise FileExistsError(f'refusing to overwrite {checkpoint}')
    with gzip.open(dataset,'rt',encoding='utf-8') as f:
        data = json.load(f)
    if tuple(data['feature_names']) != FEATURES:
        raise ValueError('dataset feature contract differs')
    training = [r for r in data['examples'] if r['split']=='train']
    validation = [r for r in data['examples'] if r['split']=='validation']
    if not training or not validation:
        raise ValueError('need existing disjoint train and validation sessions')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    network = build_network()
    wrapper = OneDTransitionRewardModel(network,target_is_delta=False,learned_rewards=False)
    train_buffer, val_buffer = buffer_for(training,seed), buffer_for(validation,seed+1)
    train_iter = BootstrapIterator(train_buffer.get_all(),batch_size=64,ensemble_size=ENSEMBLE,
                                   shuffle_each_epoch=True,rng=np.random.default_rng(seed))
    val_iter = TransitionIterator(val_buffer.get_all(),batch_size=64)
    trainer = ModelTrainer(wrapper,optim_lr=0.001)
    losses, scores = trainer.train(train_iter,dataset_val=val_iter,num_epochs=epochs,
                                   improvement_threshold=0.,silent=True)
    if not all(math.isfinite(x) for x in losses+scores):
        raise ValueError('nonfinite training loss')
    metadata = {'schema':1,'mbrl_commit':'3f93cccfc8d635f74e335a2f07aab6e9a48fc021',
        'feature_names':list(FEATURES),'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'source_manifest_sha256':data['source_manifest_sha256'],'seed':seed,'epochs':epochs,
        'train_sessions':len(training),'validation_sessions':len(validation),
        'excluded':data['counts'],'ensemble_size':ENSEMBLE,'hidden':HIDDEN,'layers':LAYERS,
        'target':'log observed completed effective service duration',
        'buffer_mapping':'zero observation + observable feature action slots; next_obs is log duration',
        'training_losses':losses,'validation_mse_log_duration':scores}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'metadata':metadata,'model_state':network.state_dict()},checkpoint)
    return metadata


class ServiceDistribution:
    def __init__(self, network):
        self.network = network.eval()
        self.forward_calls = 0

    @classmethod
    def load(cls,path):
        saved = torch.load(path,map_location='cpu',weights_only=True)
        meta = saved['metadata']
        if (meta['feature_names'] != list(FEATURES) or meta['ensemble_size'] != ENSEMBLE or
                meta['mbrl_commit'] != '3f93cccfc8d635f74e335a2f07aab6e9a48fc021'):
            raise ValueError('checkpoint model contract mismatch')
        network = build_network()
        network.load_state_dict(saved['model_state'])
        return cls(network)

    def parameters(self, features):
        x = torch.tensor([[0.,*features]],dtype=torch.float32)
        with torch.inference_mode():
            mean, logvar = self.network(x)
        self.forward_calls += 1
        values = [(float(m),float(v)) for m,v in zip(mean.flatten(),logvar.flatten())]
        if len(values)!=ENSEMBLE or any(not math.isfinite(v) for pair in values for v in pair):
            raise ValueError('invalid GaussianMLP output')
        return values

    @staticmethod
    def survival(parameters, active_ticks):
        if active_ticks <= 0:
            return 1.
        x = math.log(active_ticks)
        return sum(.5*math.erfc((x-mu)/math.sqrt(2*math.exp(logvar)))
                   for mu,logvar in parameters)/len(parameters)

    @classmethod
    def finish_probability(cls, parameters, elapsed):
        if elapsed < 0:
            raise ValueError('elapsed must be nonnegative')
        before, after = cls.survival(parameters,elapsed), cls.survival(parameters,elapsed+1)
        if before <= 1e-12:
            return 1.0
        return min(1.,max(0.,(before-after)/before))

    @classmethod
    def expected_remaining_steps(cls, parameters, elapsed, cap=60):
        """E[ceil(T)-elapsed | T>elapsed], bounded at cap active steps."""
        if elapsed < 0 or cap < 1:
            raise ValueError('invalid conditional service horizon')
        survival = cls.survival(parameters,elapsed)
        if survival <= 1e-12:
            return 1.0
        return sum(cls.survival(parameters,elapsed+k)/survival for k in range(cap))

    @staticmethod
    def mean(parameters):
        return sum(math.exp(mu+.5*math.exp(logvar)) for mu,logvar in parameters)/len(parameters)

    @staticmethod
    def nll(parameters, duration):
        if duration <= 0:
            raise ValueError('completed duration must be positive')
        x = math.log(duration)
        density = sum(math.exp(-.5*(x-mu)**2/math.exp(lv)) /
                      (duration*math.sqrt(2*math.pi*math.exp(lv))) for mu,lv in parameters)/len(parameters)
        return -math.log(max(density,1e-300))


class PETSRuleDynamics:
    """Sample neural completion only; reuse ProgressAwareRule for legal dynamics."""
    def __init__(self, geometry, distribution, seed=0):
        self.rule = ProgressAwareRule(geometry)
        self.distribution = distribution
        self.rng = random.Random(seed)
        self.unknown_progress = 0

    def reset_seed(self,seed):
        self.rng.seed(seed)

    def estimated_service_work(self, initial, terminal):
        """Expected remaining active service for known tasks, including arrivals.

        Uses forecast/observable state only. At a station with known progress,
        condition on survival; otherwise progress is explicitly unknown/zero.
        No unobserved future outage or service countdown is consulted.
        """
        known = {s.id for s in initial.samples}
        stations = {s.id:s for s in terminal.stations}
        robots = {r.id:r for r in terminal.robots}
        total = 0.0
        for sample in terminal.samples:
            if sample.id not in known or sample.status == 'completed':
                continue
            robot = robots.get(sample.robot_id)
            phases = (('to_dropoff',sample.dropoff),) if sample.status == 'in_transit' else (
                ('to_pickup',sample.pickup),('to_dropoff',sample.dropoff))
            for phase,station_id in phases:
                station = stations[station_id]
                elapsed = 0
                if (robot is not None and robot.position == station.position and
                        robot.phase == phase and robot.service_active_elapsed is not None):
                    elapsed = robot.service_active_elapsed
                params = self.distribution.parameters(service_features(terminal,robot,sample,phase,station))
                total += self.distribution.expected_remaining_steps(params,elapsed)
        return total

    def predict_next(self, observation, assignments):
        if not isinstance(observation,ObservationV2):
            observation = make_observer().observe(observation)
        assignments = tuple(assignments)
        intents = _intent(observation,assignments)
        samples = {s.id:s for s in observation.samples}
        stations = {s.id:s for s in observation.stations}
        overrides = {}
        for robot in observation.robots:
            phase,sid = intents[robot.id]
            if sid is None:
                continue
            sample = samples[sid]
            station = stations[sample.pickup if phase == 'to_pickup' else sample.dropoff]
            if robot.position != station.position or not station.available:
                continue
            progress = robot.service_active_elapsed
            if progress is None:
                if robot.service_elapsed is not None:
                    self.unknown_progress += 1
                progress = 0
            params = self.distribution.parameters(service_features(observation,robot,sample,phase,station))
            advance = self.rng.random() < self.distribution.finish_probability(params,progress)
            overrides[station.id] = progress + (1 if advance else 2)
        adjusted = replace(observation,stations=tuple(replace(s,
            expected_service_ticks=overrides[s.id],observed_effective_service_mean=float(overrides[s.id]))
            if s.id in overrides else s for s in observation.stations))
        result = self.rule.predict_next(adjusted,assignments)
        old = {s.id:s for s in observation.stations}
        return replace(result,stations=tuple(replace(s,expected_service_ticks=old[s.id].expected_service_ticks,
            observed_effective_service_mean=old[s.id].observed_effective_service_mean) for s in result.stations))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--epochs',type=int,default=35)
    args = p.parse_args()
    print(json.dumps(train(args.dataset,args.checkpoint,seed=args.seed,epochs=args.epochs)))


if __name__ == '__main__':
    main()
