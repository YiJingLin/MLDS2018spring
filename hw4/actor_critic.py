import argparse, gym, os, pickle
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=666, metavar='N',
                    help='random seed (default: 666)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--reward_threshold', type=int, default=5, 
                    help='target reward for each episode')
parser.add_argument('--n_episode', type=int, default=2000, metavar='G',
                    help='total episode')
parser.add_argument('--n_step', type=float, default=10000, metavar='G',
                    help='# of step for each episode')
parser.add_argument('--history_path', type=str, default='./hw4/log/', 
                    help='save learning record path')
parser.add_argument('--history_name', type=str, default='history_ac.pkl', 
                    help='learning history name')
    
args = parser.parse_args()


env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    prepro 210x160x3 into 80x80 
    """
    o = o[35:195]
    o = o[::2, ::2, 0]
    o[o == 144] = 0
    o[o == 109] = 0
    o[o != 0 ] = 1
    o = np.expand_dims(o, axis=0)
    return o.astype(np.float)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.action_head = nn.Linear(100*16, 3)# action 1 = 不動, action 2 = 向上, action 3 = 向下
        self.value_head = nn.Linear(100*16, 1)

        self.saved_actions = []
        self.rewards = []
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.data[0]


def select_action(state):
    # print(state.shape)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    history = []
    for i_episode in range(args.n_episode+1):
        running_reward = None

        state = env.reset()
        reward_sum = 0
        for t in range(args.n_step):  # Don't infinite loop while learning
            state = prepro(state)
            action = select_action(state)
            action = action +1
            state, reward, done, _ = env.step(action)
            reward_sum = reward_sum +reward
            model.rewards.append(reward)
            if done:
                break

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        history.append(reward_sum)
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        # if running_reward > args.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break
        if i_episode % 100 ==0:
            with open(os.path.join(args.history_path, args.history_name),'wb') as file:
                    pickle.dump(history, file)
            print('save history ...')


if __name__ == '__main__':
    main()