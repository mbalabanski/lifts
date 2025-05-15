from collections import namedtuple
from itertools import count
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import multiprocessing

import lifts
import lifts.filters

is_fork = multiprocessing.get_start_method() == "fork"

device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

lr = 3e-3

gamma = 0.99

MAX_CONTROL_INPUT = 2.5

filters = [
    # lifts.filters.GaussianNoise(0.1)
]

env = gymnasium.make('lifts/QuadRotor-v0', xml_path="./lifts/assets/quadrotor.xml", render_mode='human', filters=filters)
SavedAction = namedtuple('SavedAction', ['dist', 'log_prob', 'value'])

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, action_dim):
        super(Policy, self).__init__()

        self.rnn_layer = nn.RNN(23, 256)

        self.actor = nn.Sequential(
            nn.LazyLinear(1024),
            nn.Tanh(),
            nn.LazyLinear(1024),
            nn.Tanh(),
            nn.LazyLinear(1024),
            nn.Tanh()
        )

        self.actor_mean_output = nn.LazyLinear(action_dim)
        self.actor_variance_output = nn.LazyLinear(action_dim)

        self.critic = nn.Sequential(
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(1)
        )
        # action & reward buffer
        self.saved_actions = []
        self.prev_actions = []
        self.rewards = []

    def _forward_action_distribution_parameters(self, actor_hidden):
        mean = self.actor_mean_output(actor_hidden)
        variance = self.actor_variance_output(actor_hidden) # returns the log of the variance
        variance = torch.exp(variance)
        
        return mean, variance

    def forward(self, x):
        """
        forward of both actor and critic
        """

        actions = self._forward_action_distribution_parameters(self.actor(x))
        state_values = self.critic(x)
        
        return actions, state_values
    
model = Policy(action_dim=4)
optimizer = optim.Adam(model.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


def select_action(state):

    # scale positions to [-1, 1]
    state["agent"][0, :] /= 5
    state["payload"][0, :] /= 5
    state["target"] /= 5

    state = np.concatenate((state["agent"].flatten(), state["payload"].flatten(), state["target"].flatten()), axis=0)
    state = torch.from_numpy(state).float()
    
    action_dist_params, state_value = model(state)

    action_dist = torch.distributions.Normal(action_dist_params[0], action_dist_params[1])

    action = action_dist.sample().flatten()
    
    # smooth action to fit space
    action = torch.tanh(action) # fits to (-1, 1)
    action = MAX_CONTROL_INPUT * (action + 1) / 2 # fits to action space

    # save to action buffer
    model.saved_actions.append(SavedAction(action_dist, log_prob=action_dist_params[1], value=state_value))

    return action.detach().numpy()


def finish_episode(gamma=0.99, gae_lambda=0.95):
    """
    Compute the returns and advantages using Generalized Advantage Estimation (GAE)
    and then perform a policy update.

    GAE paper: https://arxiv.org/pdf/1506.02438
    """
    saved_actions = model.saved_actions  # SavedAction tuples (dist, log_prob, value)
    rewards = model.rewards

    values = [action.value.item() for action in saved_actions]
    
    values.append(0)

    # advantages and returns.
    gae = 0
    advantages = []
    returns = []

    # https://arxiv.org/pdf/1506.02438
    for t in reversed(range(len(rewards))):

        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

        returns.insert(0, gae + values[t])

    # create tensors from advantages and returns
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # compute the policy loss and the value loss.
    policy_losses = []
    value_losses = []

    for (saved_action, R, advantage) in zip(saved_actions, returns, advantages):
        policy_losses.append(-saved_action.log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(saved_action.value, torch.tensor([R])))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    # free rewards and saved_actions
    del model.rewards[:]
    del model.saved_actions[:]

def main():
    running_reward = 0
    
    # run infinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):
            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % 100 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        

if __name__ == '__main__':

    use_existing_model = False
    model_path = './models/actor_critic.pkl'

    if use_existing_model:
        model = torch.load(model_path, weights_only=False)
        model.eval()

    try:
        main()
    except KeyboardInterrupt:
        pass

    torch.save(model, model_path)