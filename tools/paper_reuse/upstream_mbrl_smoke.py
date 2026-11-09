"""Exercise unmodified MBRL-Lib ensemble, replay buffer and trainer on toy data."""
import json
import numpy as np
import torch
from mbrl.models import GaussianMLP, ModelTrainer, OneDTransitionRewardModel
from mbrl.util.replay_buffer import BootstrapIterator, ReplayBuffer

torch.set_num_threads(1)
torch.manual_seed(7)
rng = np.random.default_rng(7)
model = GaussianMLP(in_size=2, out_size=1, device='cpu', num_layers=2,
                    ensemble_size=3, hid_size=16, deterministic=False)
wrapped = OneDTransitionRewardModel(model, target_is_delta=False, learned_rewards=False)
buffer = ReplayBuffer(64, (1,), (1,), rng=rng)
for i in range(64):
    x = np.array([i / 64], dtype=np.float32)
    action = np.array([0.25], dtype=np.float32)
    buffer.add(x, action, x + action, 0., False, False)
iterator = BootstrapIterator(buffer.get_all(), batch_size=16, ensemble_size=3, rng=rng)
mean, logvar = model(torch.zeros(4, 2))
assert tuple(mean.shape) == (3, 4, 1) and tuple(logvar.shape) == (3, 4, 1)
old = {k:v.clone() for k,v in model.state_dict().items()}
losses, _ = ModelTrainer(wrapped, optim_lr=0.001).train(iterator, num_epochs=3, evaluate=False, silent=True)
changed = any(not torch.equal(old[k], value) for k,value in model.state_dict().items())
assert changed and len(losses) == 3 and all(np.isfinite(losses))
print(json.dumps({'source':'unmodified mbrl.models.GaussianMLP', 'device':'cpu',
                  'ensemble_size':3, 'output_shape':list(mean.shape), 'replay_size':len(buffer),
                  'trainer':'mbrl.models.ModelTrainer', 'losses':losses, 'parameters_updated':changed}))
