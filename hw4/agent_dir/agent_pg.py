from agent_dir.agent import Agent
import scipy
import numpy as np
import pickle
import gym, os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

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
        self.out = nn.Linear(100*16, 3) # action 1 = 不動, action 2 = 向上, action 3 = 向下

        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return F.softmax(self.out(x.view(x.size(0), -1)))


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.policy = Policy()
        self.args = args

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            with open(os.path.join(args.weight_path, args.model_name), 'rb') as file:
                self.policy.load_state_dict(pickle.load(file)) 
        elif args.train_pg:
            if not os.path.exists(args.weight_path):
                os.makedirs(args.weight_path)
            if not os.path.exists(args.history_path):
                os.makedirs(args.history_path)

        self.init_game_setting()


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.running_reward = None
        self.reward_sum = 0

        self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)


    def train(self, n_episode=None):
        """
        Implement your training algorithm here
        """
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.args.learning_rate, weight_decay=self.args.decay_rate)

        history = [] # record

        if n_episode is None:
            n_episode = self.args.n_episode


        for i_episode in range(n_episode+1):
            obs = self.env.reset()
            for t in range(self.args.n_step):
                action = self.make_action(obs)
                # output : 0, 1, 2
                # gym : action 1 = hold, action 2 = up, action 3 = down
                obs, reward, done, _ = self.env.step(action)
                self.reward_sum += reward
                
                self.policy.rewards.append(reward)
                if done:
                    # tracking log
                    self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
                    history.append(self.reward_sum)
                    self.reward_sum = 0
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
            self.finish_episode()

            # # Save model in every 50 episode
            # if i_episode % 50 == 0:
            #     print('ep %d: model saving...' % (i_episode))
            #     torch.save(policy.state_dict(), 'pg_params.pkl')

            if (i_episode) % 100 == 0 :
                with open(os.path.join(self.args.history_path, self.args.history_name),'wb') as file:
                    pickle.dump(history, file)
                with open(os.path.join(self.args.weight_path, self.args.model_name), 'wb') as file:
                    pickle.dump(self.policy.state_dict(), file)
                print('save history and model ...')

    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []

        if self.args.variance_reduction :
            for r in self.policy.rewards[::-1]:
                R = r + self.args.gamma * R
                rewards.insert(0, R)
            # turn rewards to pytorch tensor and standardize
            rewards = torch.Tensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 10e-8) #np.finfo(np.float32).eps)
        else:
            rewards = self.policy.rewards

        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)


        # 清理optimizer的gradient是PyTorch制式動作，去他們官網學習一下即可
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # clean rewards and saved_actions
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        # print(observation.shape)
        observation = prepro(observation)
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self.policy(Variable(observation))
        m = Categorical(probs)
        action = m.sample() # 從multinomial分佈中抽樣
        self.policy.saved_log_probs.append(m.log_prob(action)) # 蒐集log action以利於backward
        action = action.data[0] + 1
        return int(action)

