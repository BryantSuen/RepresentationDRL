import copy
from collections import namedtuple
from itertools import count
import math
from pickle import TRUE
import random
import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import gym

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from bisim.bisim_agent import BisimAgent

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_START - (EPS_START - EPS_END) * (steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model(representAgent:BisimAgent):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor[1, 4, 84, 84])
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')

    # TODO: Notice: replace none to zero, as update encoder will cause mismatch if non_final_next_states is used.
    zero_final_next_states = torch.cat([s if s is not None else torch.zeros([1, 4, 84, 84], dtype=torch.float) 
    for s in batch.next_state]).to('cuda')
    
    # state_batch: torch[BATCH_SIZE, 4, 84, 84]
    # action_batch: torch[BATCH_SIZE, 1]
    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)


    state_batch_represented = representAgent(state_batch)
    non_final_next_states_represented = representAgent(non_final_next_states)

    #if np.random.rand()<=0.5:
    state_action_values = policy_net(state_batch_represented).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values = target_net(non_final_next_states)
    next_state_values[non_final_mask]=target_net(non_final_next_states_represented).gather(1, torch.max(policy_net(non_final_next_states_represented), 1)[1].unsqueeze(1)).squeeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.huber_loss(state_action_values, expected_state_action_values.unsqueeze(1), delta=1)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    representAgent.update(state_batch, action_batch, reward_batch, zero_final_next_states)
    '''else:
        state_action_values = target_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # next_state_values = target_net(non_final_next_states)
        next_state_values[non_final_mask] = policy_net(non_final_next_states).gather(1, torch.max(
            target_net(non_final_next_states), 1)[1].unsqueeze(1)).squeeze(1)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.huber_loss(state_action_values, expected_state_action_values.unsqueeze(1), delta=1)
        optimizer.zero_grad()
        loss.backward()
        for param in target_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()'''


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = state / 255.
    # shape: [1, 4, 84, 84]
    return state.unsqueeze(0)



def train(env, n_episodes, representAgent:BisimAgent, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        # shape: [84, 84, 4]
        state = get_state(obs)
        state_represented = representAgent(state)

        total_reward = 0.0
        for t in count():
            action = select_action(state_represented)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state
            if state is not None:
                state_represented=representAgent(state)
            if steps_done > INITIAL_MEMORY:
                optimize_model(representAgent)
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
        print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(steps_done, episode, total_reward))
        if (episode+1) % (n_episodes / N_EVAL) == 0:
            global avg
            avg.append(test(env, 20, policy_net, representAgent, None, render=False, rec=False))
            #print(avg)
        if episode % (n_episodes // 3) == 0:
            torch.save(policy_net, "./results/models/double_{}_model".format(episode // (n_episodes // 3)))
            representAgent.save("./results/bisim_models/double_{}_model".format(episode // (n_episodes // 3)))

    env.close()
    return


def test(env, n_episodes, policy, representAgent:BisimAgent, path=None, render=True, rec=True):
    if rec:
        env = gym.wrappers.Monitor(env, path, force = True)
    avg_20 = 0
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        state_represented = representAgent(state)
        total_reward = 0.0
        for t in count():
            action = policy(state_represented.to('cuda')).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state
            if state is not None:
                state_represented=representAgent(state)

            if done:
                #print("Eval Finished Episode {} with reward {}".format(episode, total_reward))
                break
        avg_20 += total_reward
        
    env.close()
    print(avg_20/n_episodes)
    return avg_20 / n_episodes

def plot(avg):
    fig = plt.figure(figsize=(14,10))
    fig = fig.add_subplot(111)
    fig.plot(avg)
    fig.set_title('double-DQN Average Reward Cuves')
    fig.set(xlabel = 'Episodes',ylabel = 'Avg Reward')
    plt.savefig('./results/images/double_curves.png')

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY = 1000000
    TARGET_UPDATE = 10000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 1000
    # MEMORY_SIZE = 1000000
    MEMORY_SIZE = 10000
    N_EPISODE = 500
    # N_EPISODE = 20000
    N_EVAL = 40
    # N_EVAL = 400

    # hyperparameters for bisim
    DISCOUNT = 0.99 # 0.99
    BISIM_COEF = 1. # 0.5
    ENCODER_LR = 1e-4
    ENCODER_WEIGHT_DECAY = 0.
    DECODER_LR = 1e-4
    DECODER_WEIGHT_DECAY = 0.

    ENCODER_FEATURE_DIM = 256 # 256
    ENCODER_N_LAYERS = 2 # only available in [2, 4, 6]  default:2
    TRANSISTION_MODEL_LAYER_WIDTH = 96  # default:5
    DECODER_LAYER_SIZE = 512

    POLICY_NET_LAYERS = 2 # default:2
    POLICY_NET_FC_SIZE = 512

    ENCODER_USE_RESNET=True

    # create environment
    # env = gym.make("SpaceInvaders-v0")
    env = gym.make("Enduro-v0")

    n_actions = env.action_space.n

    env = make_env(env, stack_frames=True, episodic_life=False, clip_rewards=False, scale=False)

    bisim = BisimAgent([4, 84, 84], n_actions, 
    discount=DISCOUNT, bisim_coef=BISIM_COEF, 
    encoder_lr=ENCODER_LR, encoder_weight_decay=ENCODER_WEIGHT_DECAY,
    decoder_lr=DECODER_LR, decoder_weight_decay=DECODER_WEIGHT_DECAY,
    encoder_feature_dim=ENCODER_FEATURE_DIM, encoder_n_layers=ENCODER_N_LAYERS,
    t_model_layer_width=TRANSISTION_MODEL_LAYER_WIDTH,
    decoder_layer_size=DECODER_LAYER_SIZE,
    useResNet=ENCODER_USE_RESNET)

    # create networks
    # policy_net = DQN(n_actions=env.action_space.n).to(device)
    # target_net = DQN(n_actions=env.action_space.n).to(device)
    policy_net = MLP(input_size=ENCODER_FEATURE_DIM, n_layers=POLICY_NET_LAYERS, fc_size=POLICY_NET_FC_SIZE, n_actions=n_actions).to(device)
    target_net = MLP(input_size=ENCODER_FEATURE_DIM, n_layers=POLICY_NET_LAYERS, fc_size=POLICY_NET_FC_SIZE, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    avg = []

    # train model
    train(env, N_EPISODE, render=False, representAgent=bisim)
    np.save('./results/arrays/double_avg.npy', avg)
    plot(avg)
    torch.save(policy_net, "./results/models/double_final_model")
    bisim.save("./results/bisim_models/double_final_model")

    policy_net = torch.load("./results/models/double_0_model")
    bisim.load("./results/bisim_models/double_0_model")
    test(env, 1, policy_net, bisim, path='./videos/' + 'double_video_0', render=False, rec=False)

    policy_net = torch.load("./results/models/double_1_model")
    bisim.load("./results/bisim_models/double_1_model")
    test(env, 1, policy_net, bisim, path='./videos/' + 'double_video_1', render=False, rec=False)

    policy_net = torch.load("./results/models/double_2_model")
    bisim.load("./results/bisim_models/double_2_model")
    test(env, 1, policy_net, bisim, path='./videos/' + 'double_video_2', render=False, rec=False)

    policy_net = torch.load("./results/models/double_final_model")
    bisim.load("./results/bisim_models/double_final_model")
    test(env, 1, policy_net, bisim, path='./videos/' + 'double_video_3', render=False, rec=False)
