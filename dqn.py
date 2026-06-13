"""
DQN implementation for LunarLander-v2.
Based on the Gymnasium DQN tutorial (https://gymnasium.farama.org/tutorials/training_agents/blackjack/).
"""
import random
import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Small MLP — thread-coordination overhead exceeds the work with the default
# thread count, and extra threads starve the Streamlit UI thread.
torch.set_num_threads(2)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SumTree:
    """Binary tree where each leaf holds a priority and each internal node holds
    the sum of its children. Enables O(log n) proportional sampling and updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)   # internal nodes + leaves
        self.data = [None] * capacity            # transitions, parallel to leaves
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):          # reached a leaf
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al. 2016).

    Transitions are sampled with probability ∝ (|TD-error| + eps)^alpha, and the
    resulting bias is corrected with importance-sampling weights scaled by beta."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = 1e-5
        self.max_priority = 1.0   # new transitions enter at the current max

    def push(self, *args):
        self.tree.add(self.max_priority ** self.alpha, Transition(*args))

    def sample(self, batch_size: int, beta: float):
        batch, idxs, priorities = [], [], []
        total = self.tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:                     # rare edge: empty slot — resample
                idx, p, data = self.tree.get(random.uniform(0, total))
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        probs = np.array(priorities) / total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()                 # normalise so max weight = 1
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = abs(float(err)) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p ** self.alpha)

    def __len__(self):
        return self.tree.n_entries


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 50_000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        hidden: int = 128,
        device: str = "cpu",
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.policy_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.use_per = use_per
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.buffer = (PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
                       if use_per else ReplayBuffer(buffer_size))
        self.steps_done = 0

    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay
        )

    @property
    def beta(self) -> float:
        """Importance-sampling exponent, annealed from per_beta_start to 1.0."""
        if self.per_beta_frames <= 0:
            return 1.0
        frac = min(1.0, self.steps_done / self.per_beta_frames)
        return self.per_beta_start + (1.0 - self.per_beta_start) * frac

    def select_action(self, obs: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(t).argmax(dim=1).item()

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor([action]),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
        )

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        if self.use_per:
            transitions, idxs, is_weights = self.buffer.sample(self.batch_size, self.beta)
            is_weights = torch.tensor(is_weights, device=self.device).unsqueeze(1)
        else:
            transitions = self.buffer.sample(self.batch_size)
            idxs, is_weights = None, None

        batch = Transition(*zip(*transitions))

        states = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.action).to(self.device)
        rewards = torch.stack(batch.reward).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.stack(batch.done).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True).values
            next_q[dones] = 0.0
            targets = rewards + self.gamma * next_q

        td_errors = q_values - targets
        elementwise = nn.SmoothL1Loss(reduction="none")(q_values, targets)
        loss = (is_weights * elementwise).mean() if self.use_per else elementwise.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        if self.use_per:
            self.buffer.update_priorities(idxs, td_errors.detach().squeeze(1).cpu().numpy())

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
