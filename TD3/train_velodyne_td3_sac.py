import os
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_memory import ReplayMemory
from velodyne_env import GazeboEnv
from sac import SAC, SAC_PATH

import itertools

from gym import spaces



# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number    
#seed = 4  # Random seed number    # 221007
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e7  # Maximum number of steps to perform   # 5e6
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
#batch_size = 40  # Size of the mini-batch
batch_size = 256  # 221007
#batch_size = 200  # 221007
#discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)   # TODO 0.99로 바꿔보기
discount = 0.99   # 221007
tau = 0.005  # Soft target update variable (should be close to 0) 
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise 
policy_freq = 2  # Frequency of Actor network updates (delayed policy updates cycle)
buffer_size = 1e6  # Maximum size of the buffer   # 1000000  as 100k
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model   
random_near_obstacle = False  # To take random actions near obstacles or not
start_timesteps = 2e3 # 221006   # https://github.com/sfujim/TD3/blob/master/main.py 
save_interval = 200

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
#parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
#parser.add_argument('--batch_size', type=int, default=256, metavar='N',
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
#parser.add_argument('--cuda', action="store_true",
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):   
    os.makedirs("./pytorch_models")

writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Create the training environment
environment_dim = 20
robot_dim = 4
human_num = 12
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim  # 24
#state_dim = environment_dim + robot_dim + human_num
action_dim = 2
max_action = 1

# Create the network

action_bound = [[0, -1], [1, 1]] 
action_bound = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
PATH_AS_INPUT = True
#PATH_AS_INPUT = False
if PATH_AS_INPUT:
    agent = SAC_PATH(state_dim, action_bound, args) 
else:
    agent = SAC(state_dim, action_bound, args)


# Create a replay buffer
memory = ReplayMemory(args.replay_size, args.seed)

ckpt_path = "checkpoints/sac_checkpoint_TD3_velodyne_200"
evaluate = False
if load_model:
        agent.load_checkpoint(ckpt_path, evaluate)
        print('잘 불러왔다.')

# Begin the training loop

total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    #print('----------------')
    #print('i_episode:',i_episode, done)

    while not done:
        if args.start_steps > total_numsteps:
            #action = env.action_space.sample()  # Sample random action   # TODO 이부분을 완전 랜덤하게 해야 한다
            action = agent.select_action(state)
            action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(-max_action, max_action)   
        else:   # 여기 실제로 되는지
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        
        
        a_in = [(action[0] + 1) / 2, action[1]]  

        #next_state, reward, done, _ = env.step(action) # Step
        #print('ain:',a_in)
        next_state, reward, done, target = env.step(a_in, episode_steps) # 221102
        print('after step:',episode_steps, total_numsteps)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        
        mask = 1 if episode_steps == max_ep else float(not done)
        
        ####done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
        
        
        done = 1 if episode_steps == max_ep else int(done)
        
        
        
        ####### 221015 unlimited loop exlude
        ######if episode_steps > 501:
        ######    done = True
         
        

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break
    
    # 221104
    status = 'NA'
    if done and target:
        status = 'Success'
    elif done and episode_steps == max_ep:
        status = 'Timeout'
    elif done:
        status = 'Collision'
    

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), status))

    if i_episode != 0 and i_episode % save_interval == 0 and save_model:
        
        agent.save_checkpoint(file_name, i_episode)
        

    evaluation_step = 100
    #if i_episode % 10 == 0 and args.eval is True:
    if i_episode % evaluation_step == 0 and i_episode != 0 and args.eval is True:
    #if i_episode % 10 == 0 and i_episode != 0 and args.eval is True:
        #print('i_episode:',i_episode)
        print('Validating...')
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            print('whileㅈㅓㄴ')
            flag = 0
            while not done:
                action = agent.select_action(state, evaluate=True)
                a_in = [(action[0] + 1) / 2, action[1]]  
                #next_state, reward, done, _ = env.step(action)
                #next_state, reward, done, _ = env.step(a_in)
                next_state, reward, done, _ = env.step(a_in, flag)
                episode_reward += reward


                state = next_state
                flag += 1
                
                if flag > 501:
                    break
            avg_reward += episode_reward
            print('while끝')
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    #print('end tien done:',done)