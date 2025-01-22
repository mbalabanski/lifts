from collections import namedtuple
from itertools import count
import gymnasium
import lifts

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import multiprocessing
from torch.distributions.normal import Normal

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000

total_frames = 50_000

sub_batch_size = 64 
num_epochs = 10  
clip_epsilon = 0.2

gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

MAX_CONTROL_INPUT = 2.0

RUN_EVAL = False

render_mode = 'human' if RUN_EVAL else 'rgb_array'


env = gymnasium.make('lifts/QuadRotor-v0', xml_path="./lifts/assets/quadrotor.xml", render_mode=render_mode)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Actor(nn.Module):
    def __init__(self, output_dim=4):
        super(Actor, self).__init__()
        self.fc = nn.LazyLinear(128)
        self.relu = nn.ReLU()
        self.mu_layer = nn.LazyLinear(output_dim)
        self.sigma_layer = nn.LazyLinear(output_dim)
        self.softplus = nn.Softplus()  # Ensures standard deviation is positive

    def forward(self, state):
        x = self.relu(self.fc(state))
        mu = torch.sigmoid(self.mu_layer(x)) * MAX_CONTROL_INPUT
        sigma = torch.sigmoid(self.sigma_layer(x))  # or you could use exp for sigma to ensure it's positive
        return mu, sigma + 0.001

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)

# Hyperparameters
learning_rate = 0.0025
num_episodes = 1000
gamma = 0.99  # Discount factor

# Model
actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

def dict_to_state(state_dict):
    return np.concatenate((state_dict["agent"], state_dict["payload"])).flatten()


def main():
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = dict_to_state(state)

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
            mu, sigma = actor(state_tensor)
            dist = Normal(mu, sigma)
            action = dist.sample()
            clipped_action = np.clip(np.array(action.flatten()), env.action_space.low[0], env.action_space.high[0])  # enforce action boundaries

            next_state, reward, done, _, _ = env.step(clipped_action)
            next_state = dict_to_state(next_state)

            total_reward += reward

            value = critic(state_tensor)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = critic(next_state_tensor)
            target_value = reward + gamma * next_value * (1 - int(done))

            advantage = target_value.detach() - value
            actor_loss = -dist.log_prob(action) * advantage
            critic_loss = advantage.pow(2)

            actor_loss = actor_loss.mean()
            critic_loss = critic_loss.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            GRADIENT_CLIP = 0.5  
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRADIENT_CLIP)

            state = next_state

        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}/{num_episodes}, Average reward: {total_reward / episode}')
            print(f"Calculated Dist Params: MU: {mu}, SIGMA: {sigma}")

if __name__ == "__main__":

    if RUN_EVAL:
        actor = torch.load("ACTOR_MODEL")
        critic = torch.load("CRITIC_MODEL")

        actor.eval()
        critic.eval()

    try:
        main()
    except KeyboardInterrupt:
        # save models
        torch.save(actor, 'ACTOR_MODEL')
        torch.save(critic, "CRITIC_MODEL")