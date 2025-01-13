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
from torch.distributions.bernoulli import Bernoulli

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

MAX_CONTROL_INPUT = 1.5


env = gymnasium.make('lifts/QuadRotor-v0', xml_path="./lifts/assets/quadrotor_x.xml", render_mode='human')


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(4)
        )

        self.critic = nn.Sequential(
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(1)
        )

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """

        # actor: choses action to take from state s_t
        # by returning probability of each action
        actions = self.actor(x)

        # critic: evaluates being in the state s_t
        state_values = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return actions, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

epsilon = 0.10

def select_action(state):

    state = np.concatenate((state["agent"], state["payload"])).flatten()

    state = torch.from_numpy(state).float()

    
    action, state_value = model(state)

    if np.random.rand() < epsilon:
        action = torch.from_numpy(env.action_space.sample()).float().flatten()

    # make sure action fits action space
    action = torch.clamp(action, 0, MAX_CONTROL_INPUT)

    # save to action buffer
    model.saved_actions.append(SavedAction(action, state_value))


    # the action to take (left or right)
    return action.detach().numpy()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + 0.9 * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

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

        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()