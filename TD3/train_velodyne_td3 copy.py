import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

from gym import spaces

PATH_AS_INPUT = False

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    avg_suc = 0.0
    avg_to = 0.0
    avg_col = 0.0
    avg_length = 0.0
    col = 0
    suc = 0  # 221005
    to = 0
    length = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:   
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            #state, reward, done, _ = env.step(a_in)
            state, reward, done, target = env.step(a_in)  
            avg_reward += reward
            count += 1
            #if reward < -90:
            if done and target != True:
                col += 1        
            if target:
                suc += 1
                length += count
            #print(target, reward)
                
        if count >= 501:
            to += 1
        print('[Evaluate]Reward:',avg_reward,'Done:',done,'Success:',suc,'Collision:',col)
    avg_reward /= eval_episodes
    avg_suc = suc / eval_episodes
    avg_to = to / eval_episodes
    avg_col = col / eval_episodes
    avg_length = length / eval_episodes
    print("..............................................")
    print(
        #"Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        "%i Evaluation Episodes, Epoch %i: Avg.Reward: %f, Avg.Collision: %f, Avg.Success: %f, Avg.Timeout: %f, Avg.length: %f"
        % (eval_episodes, epoch, avg_reward, avg_col, avg_suc, avg_to, avg_length)
    )
    print("..............................................")
    return avg_reward


writer = SummaryWriter()

# ref: https://github.com/sfujim/TD3/blob/master/TD3.py

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        if PATH_AS_INPUT:
            self.layer_0_traj = nn.Linear(10, 400)
            self.layer_0_s = nn.Linear(state_dim,400)
            self.layer_1 = nn.Linear(800, 800)
        else:
            self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        if PATH_AS_INPUT:
            traj = s[:,-10:]
            state = s[:,:24]
            traj = F.relu(self.layer_0_traj(traj))    # 400
            state = F.relu(self.layer_0_s(state))     # 400
            s = torch.cat((state, traj), dim=-1)      # 800
            s = F.relu(self.layer_1(s))
            s = F.relu(self.layer_2(s))
            a = self.tanh(self.layer_3(s))
        
        else:
            s = F.relu(self.layer_1(s))
            s = F.relu(self.layer_2(s))
            a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        if PATH_AS_INPUT:
            self.layer_0_traj_A = nn.Linear(10, 400)
            self.layer_0_s_A = nn.Linear(state_dim,400)
            self.layer_1 = nn.Linear(800, 800)
            self.layer_0_traj_B = nn.Linear(10, 400)
            self.layer_0_s_B = nn.Linear(state_dim,400)
            self.layer_4 = nn.Linear(800, 800)
        else:
            self.layer_1 = nn.Linear(state_dim, 800)
            self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        if PATH_AS_INPUT:
            traj = s[:,-10:]
            state = s[:,:24]
            traj1 = F.relu(self.layer_0_traj_A(traj))
            state1 = F.relu(self.layer_0_s_A(state))
            s1 = torch.cat((state1, traj1), dim = -1)
            s1 = F.relu(self.layer_1(s1))
            self.layer_2_s(s1)
            self.layer_2_a(a)
            s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
            s12 = torch.mm(a, self.layer_2_a.weight.data.t())
            s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
            q1 = self.layer_3(s1)
            
            traj2 = F.relu(self.layer_0_traj_B(traj))
            state2 = F.relu(self.layer_0_s_B(state))
            s2 = torch.cat((state2, traj2), dim = -1)
            s2 = F.relu(self.layer_4(s2))
            self.layer_5_s(s2)
            self.layer_5_a(a)
            s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
            s22 = torch.mm(a, self.layer_5_a.weight.data.t())
            s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
            q2 = self.layer_6(s2)
            
            pass
        else:
            s1 = F.relu(self.layer_1(s))
            self.layer_2_s(s1)
            self.layer_2_a(a)
            s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
            s12 = torch.mm(a, self.layer_2_a.weight.data.t())
            s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
            q1 = self.layer_3(s1)

            s2 = F.relu(self.layer_4(s))
            self.layer_5_s(s2)
            self.layer_5_a(a)
            s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
            s22 = torch.mm(a, self.layer_5_a.weight.data.t())
            s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
            q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
#seed = 0  # Random seed number    
seed = 2  # Random seed number    # 221007
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
load_model = True  # Weather to load a stored model   
random_near_obstacle = False  # To take random actions near obstacles or not
start_timesteps = 2e3 # 221006   # https://github.com/sfujim/TD3/blob/master/main.py 

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):   
    os.makedirs("./pytorch_models")

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
network = TD3(state_dim, action_dim, max_action) ##221006

# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

updates = 0
episode_reward = 0 # 221006
rewards = []

# Begin the training loop
while timestep < max_timesteps:   # < 5000000

    # On termination of episode
    if done:
        ############# Train #################33
        if timestep != 0:        
            
            network.train(
                replay_buffer,
                episode_timesteps,   
                batch_size,    # 40.
                discount,      # 0.99999   
                tau,           # 0.005
                policy_noise,  # 0.5
                noise_clip,    # 0.5
                policy_freq,   # 2
            )
            
        #########221006 episode reward 처리 ###############(지워도 됨)
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:])
        writer.add_scalar("Avg.reward:",avg_reward,timestep)
        writer.add_scalar("Episode reward:",episode_reward, timestep)
        
        ###########3 Evaluate episode ################3
        if timesteps_since_eval >= eval_freq:   # > 5000
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)    
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        print('[',timestep,']Reward:',episode_reward,'Done:',done)

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:   # 1 > 0.1
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)   # 1- (0.9/500000)
    writer.add_scalar("expl_noise:",expl_noise, timestep)

    action = network.get_action(np.array(state))
    
    
    ### 1. Diminishing explorration noise
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action)   
    
    ### 2. fixed exploration noise
    
    '''
    action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(
        -max_action, max_action)   
    '''




    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_reward += reward
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)