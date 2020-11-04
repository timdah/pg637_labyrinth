import torch
from copy import deepcopy
import random
from src import environment as env
import math
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_STATES = len(env.position_ids)
NUM_ACTIONS = 4


class DQN(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super(DQN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_actions),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        state, action, reward, next_state, done = tuple(map(list, zip(*random.sample(self.buffer, batch_size))))
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.tensor(reward, device=device)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.uint8, device=device)
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
        tensor = torch.zeros(NUM_STATES, device=device)
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
    def step(cls, state: int, action: int) -> tuple:
        action = cls.get_action_string(action)
        new_state, reward = env.move(action, state)
        done = reward == -1 or reward == 1
        return new_state, reward, done


class Agent:
    def __init__(self, q_network: DQN, target_q_network: DQN):
        self.q_network = q_network
        self.target_q_network = target_q_network

    def act(self, state: int, epsilon: float):
        if random.random() > epsilon:
            state_tensor = Utils.get_one_hot_tensor(state)
            q_value = self.q_network.forward(state_tensor.unsqueeze(0))
            return q_value.argmax()
        return torch.tensor(random.randrange(NUM_ACTIONS))


def gradient_descent_step(agent: Agent, optimizer: torch.optim.Optimizer, replay_buffer: ReplayBuffer, batch_size: int, gamma: float):
    # sample random minibatch
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # compute Yj (expected q_value)
    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    target_next_q_values = agent.target_q_network(next_state)
    max_indices = torch.max(target_next_q_values, dim=1)
    target_next_q_value = target_next_q_values.gather(1, max_indices.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * target_next_q_value * (1 - done)

    # compute loss
    loss = (expected_q_value - q_value).pow(2).mean()

    # update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def run_dqn(lr: float, rb_size: int):
    q_network = DQN(NUM_STATES, NUM_ACTIONS).to(device)
    target_q_network = deepcopy(q_network).to(device)
    agent = Agent(q_network, target_q_network)
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(rb_size)

    state = env.entry_id
    action = agent.act(state, epsilon=0)
    print(action)
    next_state, reward, done = Utils.step(state, action)
    state = Utils.get_one_hot_tensor(state)
    next_state = Utils.get_one_hot_tensor(next_state)
    print((state, action, reward, next_state, done))
    replay_buffer.push(state, action, reward, next_state, done)
    replay_buffer.push(state, action, reward, next_state, done)
    state, action, reward, next_state, done = replay_buffer.sample(batch_size=2)
    print((state, action, reward, next_state, done))
    print(state.size())
    print(action.size())
    print(reward.size())
    print(next_state.size())
    print(done.size())
    print(done)


if __name__ == "__main__":
    frames = 10000
    final_frames = 10000
    epsilon_start = 1.0
    epsilon_final = 0.1
    buffer_size = 10
    learn_rate = 0.1

    # print(Utils.get_epsilon(epsilon_start, epsilon_final, final_frames, 1000))
    # print(Utils.get_epsilon(epsilon_start, epsilon_final, final_frames, 5000))

    run_dqn(lr=learn_rate, rb_size=buffer_size)
