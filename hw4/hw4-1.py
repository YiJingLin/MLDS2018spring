import os
import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=5, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--n_episode', type=int, default=10000, metavar='N',
                    help='n_episode (default: 10000)')
args = parser.parse_args()


env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


def prepro(I):
    """ prepro 210x160x3 into 6400 """
    # crop
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    I = np.expand_dims(I, axis=0)
    return I.astype(np.float)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.out = nn.Linear(100*16, 3) # action 1 = 不動, action 2 = 向上, action 3 = 向下
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return F.softmax(self.out(x.view(x.size(0), -1)))


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    m = Categorical(probs)
    action = m.sample() # 從multinomial分佈中抽樣
    policy.saved_log_probs.append(m.log_prob(action)) # 蒐集log action以利於backward
    return action.data[0]


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 10e-8) #np.finfo(np.float32).eps)
    
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    # 清理optimizer的gradient是PyTorch制式動作，去他們官網學習一下即可
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]

    

# Main loop
running_reward = None
reward_sum = 0


# plt.show()

# built policy network
policy = Policy()


# check & load pretrain model
# if os.path.isfile('pg_params.pkl'):
#     print('Load Policy Network parametets ...')
#     policy.load_state_dict(torch.load('pg_params.pkl'))


# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

history = []

for i_episode in range(2000):
    state = env.reset()
    for t in range(10000):
        state = prepro(state)
        action = select_action(state)
        # output : 0, 1, 2
        # gym : action 1 = hold, action 2 = up, action 3 = down
        action = action + 1
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        
        policy.rewards.append(reward)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            history.append(reward_sum)
            reward_sum = 0
            break
        
        # if reward != 0:
            # print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

        # if t%30==0:
        #     plt.imshow(state)  # plot the episode vt
        #     plt.draw()
        #     plt.pause(0.0001)

    # use policy gradient update model weights
    # if i_episode % args.batch_size == 0 :
    print('ep %d: policy network parameters updating...' % (i_episode))
    finish_episode()

    # # Save model in every 50 episode
    # if i_episode % 50 == 0:
    #     print('ep %d: model saving...' % (i_episode))
    #     torch.save(policy.state_dict(), 'pg_params.pkl')

with open('history.pkl','wb') as file:
    pickle.dump(history, file)