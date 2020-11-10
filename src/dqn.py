import sys

import gym
import torch
from copy import deepcopy
import random
from src import environment as env_lab
from collections import deque

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_STATES = len(env_lab.position_ids)
NUM_ACTIONS = 4


class DQN(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super(DQN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, 64),
            torch.nn.ReLU(),
            # torch.nn.Linear(64, 64),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, num_actions),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: float, reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        state, action, reward, next_state, done = tuple(map(list, zip(*random.sample(self.buffer, batch_size))))
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.tensor(reward, device=DEVICE)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.uint8, device=DEVICE)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Utils:
    actions = ['left', 'right', 'up', 'down']

    @classmethod
    def get_action_string(cls, action: int) -> str:
        return cls.actions[action]

    @classmethod
    def get_action_number(cls, action: str) -> int:
        return cls.actions.index(action)

    @staticmethod
    def get_one_hot_tensor(state: int):
        tensor = torch.zeros(NUM_STATES, device=DEVICE)
        tensor[state] = 1
        return tensor

    @staticmethod
    def get_epsilon(epsilon_start, epsilon_final, final_frame, current_frame):
        return epsilon_start - (epsilon_start - epsilon_final) * (current_frame / final_frame)

    @staticmethod
    def hard_update(q_network: DQN, target_q_network: DQN):
        for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
            if t_param is param:
                continue
            new_param = param.data
            t_param.data.copy_(new_param)

    @classmethod
    def step_labyrinth(cls, state: int, action: int) -> tuple:
        action = cls.get_action_string(action)
        new_state, reward = env_lab.move(action, state)
        done = reward == -1 or reward == 1
        return new_state, reward, done

    @classmethod
    def extract_deterministic_policy(cls, target_q_network: DQN) -> tuple:
        """
        Extracts a greedy deterministic policy from a Q-Network
        :param target_q_network: Network to extract policy from
        :return: policy pi, state values under policy pi
        """
        with torch.no_grad():
            state_tensors = [Utils.get_one_hot_tensor(i) for i in range(NUM_STATES)]
            state_tensors = torch.stack(state_tensors)
            q_values = target_q_network.forward(state_tensors)
            max_indices = torch.max(q_values, dim=1)
            extracted_policy = [cls.get_action_string(action) for action in max_indices.indices]
            state_values_under_extracted_policy = list(max_indices.values)
            return extracted_policy, state_values_under_extracted_policy

    @staticmethod
    def progress(frame, max_frames, last_reward: float, last_loss: float, wins=None, loses=None, steps_per_episode: int = None):
        bar_len = 60
        filled_len = int(round(bar_len * frame / float(max_frames)))

        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        finish = ' - Done!\r\n' if frame == max_frames else ''
        win_stats = f' - Wins: {wins} / Loses: {loses}' if wins is not None else ''
        if steps_per_episode:
            last_episode = f' - Last Episode: (steps = {steps_per_episode:4}, reward = {last_reward:.4f}, loss = {last_loss:.4f})'
        else:
            last_episode = f' - Reward = {last_reward:.4f}, Loss = {last_loss:.4f}'

        sys.stdout.write(
            f'\r[{bar}] {frame}/{max_frames} Frames{win_stats}{last_episode}{finish}')
        sys.stdout.flush()


class AgentLabyrinth:
    def __init__(self, q_network: DQN, target_q_network: DQN):
        self.q_network = q_network
        self.target_q_network = target_q_network

    def act(self, state: int, epsilon: float):
        if random.random() > epsilon:
            state_tensor = Utils.get_one_hot_tensor(state)
            q_value = self.q_network.forward(state_tensor.unsqueeze(0))
            return q_value.argmax()
        return torch.tensor(random.randrange(NUM_ACTIONS), device=DEVICE)


class AgentGym:
    def __init__(self, env, q_network: DQN, target_q_network: DQN):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            q_value = self.q_network.forward(state_tensor.unsqueeze(0))
            return q_value.argmax()
        return torch.tensor(random.randrange(self.num_actions), device=DEVICE)


def gradient_descent_step(agent, optimizer: torch.optim.Optimizer,
                          replay_buffer: ReplayBuffer, batch_size: int, gamma: float) -> torch.Tensor:
    # sample random mini-batch
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # compute y_j (expected q_value)
    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    target_next_q_values = agent.target_q_network(next_state)
    max_indices = torch.max(target_next_q_values, dim=1).indices
    target_next_q_value = target_next_q_values.gather(1, max_indices.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * target_next_q_value * (1 - done)

    # compute loss
    # loss = criterion(q_value, expected_q_value)
    loss = (expected_q_value - q_value).pow(2).mean()

    # update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_dqn_gym(env_name: str, lr: float, rb_size: int, max_frames: int, start_train_frame: int,
                  epsilon_start: float, epsilon_end: float, epsilon_decay: int,
                  batch_size: int, gamma: float, target_network_update_freq: int, log_every: int):
    """
    Train a DQN to solve the labyrinth
    :param env_name: Gym environment to create
    :param lr: Learning rate
    :param rb_size: Replay buffer size
    :param max_frames: Maximum steps taken
    :param start_train_frame: Time at which the training of the DQN starts
    :param epsilon_start: Initial value of epsilon
    :param epsilon_end: Target value of epsilon
    :param epsilon_decay: Timestep at which epsilon_end is reached and freezes
    :param batch_size: Batch size to train the network with
    :param gamma: Discount factor of the rewards
    :param target_network_update_freq: Interval to update the target network
    :param log_every: Log after every x steps
    :return:
    """
    env = gym.make(env_name)

    q_network = DQN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    target_q_network = deepcopy(q_network).to(DEVICE)
    agent = AgentGym(env, q_network, target_q_network)
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    # criterion = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(rb_size)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()
    env.render()

    for frame in range(1, max_frames + 1):
        # With probability epsilon select a random action a_t
        # otherwise select a_t = argmax Q(s_t, a)
        epsilon = Utils.get_epsilon(epsilon_start, epsilon_end, epsilon_decay, frame)
        action = agent.act(state, epsilon)

        # Execute action a_t in environment and observe reward r_t
        next_state, reward, done, _ = env.step(action=int(action.cpu()))
        env.render()

        # Store transition (s_t, a_t, r_t, s_t+1) in replay buffer
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
        replay_buffer.push(state=state_tensor, action=action, reward=reward, next_state=next_state_tensor, done=done)

        state = next_state
        episode_reward += reward

        # End of an episode
        if done:
            state = env.reset()
            env.render()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > start_train_frame:
            # Perform SGD step
            # loss = gradient_descent_step(agent, optimizer, criterion, replay_buffer, batch_size, gamma)
            loss = gradient_descent_step(agent, optimizer, replay_buffer, batch_size, gamma)
            losses.append(loss.data)

            # Update the target network every target_network_update_freq steps
            if frame % target_network_update_freq == 0:
                Utils.hard_update(q_network, target_q_network)

        # Logging
        if frame % log_every == 0:
            last_reward = all_rewards[-1] if len(all_rewards) > 0 else 0
            last_loss = losses[-1] if len(losses) > 0 else 0
            Utils.progress(frame, max_frames, last_reward, last_loss)
    env.close()


def train_dqn_lab(lr: float, rb_size: int, max_frames: int, start_train_frame: int,
                  epsilon_start: float, epsilon_end: float, epsilon_decay: int,
                  batch_size: int, gamma: float, target_network_update_freq: int, log_every: int):
    """
    Train a DQN to solve the labyrinth
    :param lr: Learning rate
    :param rb_size: Replay buffer size
    :param max_frames: Maximum steps taken
    :param start_train_frame: Time at which the training of the DQN starts
    :param epsilon_start: Initial value of epsilon
    :param epsilon_end: Target value of epsilon
    :param epsilon_decay: Timestep at which epsilon_end is reached and freezes
    :param batch_size: Batch size to train the network with
    :param gamma: Discount factor of the rewards
    :param target_network_update_freq: Interval to update the target network
    :param log_every: Log after every x steps
    :return:
    """
    q_network = DQN(NUM_STATES, NUM_ACTIONS).to(DEVICE)
    target_q_network = deepcopy(q_network).to(DEVICE)
    agent = AgentLabyrinth(q_network, target_q_network)
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(rb_size)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env_lab.entry_id

    # Stats for logging
    wins = loses = steps_per_episode = steps = 0
    states_visited = [0] * NUM_STATES

    for frame in range(1, max_frames + 1):
        # With probability epsilon select a random action a_t
        # otherwise select a_t = argmax Q(s_t, a)
        epsilon = Utils.get_epsilon(epsilon_start, epsilon_end, epsilon_decay, frame)
        action = agent.act(state, epsilon)

        # Execute action a_t in environment and observe reward r_t
        next_state, reward, done = Utils.step_labyrinth(state, action)

        # Store transition (s_t, a_t, r_t, s_t+1) in replay buffer
        replay_buffer.push(state=Utils.get_one_hot_tensor(state), action=action, reward=reward,
                           next_state=Utils.get_one_hot_tensor(next_state), done=done)

        state = next_state
        episode_reward += reward
        steps += 1
        states_visited[state] += 1

        # End of an episode
        if done:
            state = env_lab.entry_id
            all_rewards.append(episode_reward)
            episode_reward = 0
            if reward == 1:
                wins += 1
            else:
                loses += 1
            steps_per_episode = steps
            steps = 0

        if len(replay_buffer) > start_train_frame:
            # Perform SGD step
            loss = gradient_descent_step(agent, optimizer, replay_buffer, batch_size, gamma)
            losses.append(loss.data)

            # Update the target network every target_network_update_freq steps
            if frame % target_network_update_freq == 0:
                Utils.hard_update(q_network, target_q_network)

        # Logging
        if frame % log_every == 0:
            last_reward = all_rewards[-1] if len(all_rewards) > 0 else 0
            last_loss = losses[-1] if len(losses) > 0 else 0
            Utils.progress(frame, max_frames, last_reward, last_loss, wins, loses, steps_per_episode)

    target_policy, state_values_policy = Utils.extract_deterministic_policy(target_q_network)
    return target_policy, state_values_policy, states_visited

# if __name__ == "__main__":
#     train_dqn_gym(env_name='CartPole-v0',
#                   lr=0.001,
#                   rb_size=5000,
#                   max_frames=10000,
#                   start_train_frame=32,
#                   epsilon_start=1.0,
#                   epsilon_end=0.01,
#                   epsilon_decay=7000,
#                   batch_size=32,
#                   gamma=0.99,
#                   target_network_update_freq=1000,
#                   log_every=100)
