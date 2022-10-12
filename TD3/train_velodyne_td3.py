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
from model_sac.sac import SAC

def evaluate(network, epoch, agent, eval_episodes=10):
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
            #action = agent.select_action(np.array(state))
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

# ref: https://github.com/sfujim/TD3/blob/master/TD3.py
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_traj = nn.Linear(10, 600)  # 221010
        #self.layer_3 = nn.Linear(600, action_dim)
        self.layer_3 = nn.Linear(600+600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        #print('original:',s, len(s), s.shape)
        traj = s[:,-10:]
        s = s[:,:24]
        #print('s:',s,len(s), s.shape)
        #print('traj:',traj, len(traj), traj.shape)
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        traj = F.relu(self.layer_traj(traj))
        s = torch.cat((s, traj), dim =-1)  # 221010
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        '''
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)
        '''
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_traj_1 = nn.Linear(10, 600)  # 221010
        #self.layer_3 = nn.Linear(600, 1)
        self.layer_3 = nn.Linear(600+600, 1)
        
        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 800)
        self.layer_5 = nn.Linear(800, 600)
        self.layer_traj_2 = nn.Linear(10, 600)  # 221010
        #self.layer_6 = nn.Linear(600, 1)
        self.layer_6 = nn.Linear(600+600, 1)

    def forward(self, s, a):
        '''
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
        '''
        traj = s[:,-10:]
        s = s[:,:24]
        
        sa  = torch.cat([s, a], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        traj_1 = F.relu(self.layer_traj_1(traj))  # 221010
        q1 = torch.cat((q1, traj_1), dim =-1)  # 221010
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        traj_2 = F.relu(self.layer_traj_2(traj))  # 221010
        q2 = torch.cat((q2, traj_2), dim =-1)  # 221010
        q2 = self.layer_6(q2)
        
        
        return q1, q2
    
    def Q1(self, s, a):   # 221010
        traj = s[:,-10:]
        s = s[:,:24]

        sa = torch.cat([s, a], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        traj_1 = F.relu(self.layer_traj_1(traj))  # 221010
        q1 = torch.cat((q1, traj_1), dim =-1)  # 221010
        q1 = self.layer_3(q1)
        return q1

writer = SummaryWriter()

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

        self.max_action = max_action   # 1
        #self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):   # ref: https://github.com/djbyrne/TD3/blob/master/TD3.ipynb
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
        ac_act_loss = 0
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

            # Add clipped noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair (Compute the target Q value)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)               # tensorboard
            max_Q = max(max_Q, torch.max(target_Q))    # tensorboard
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters  (Get current Q estimates)
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value  (Compute critic loss)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent  (Optimize the critic)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:    # 2
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)

                # Compute actor loss
                '''
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()                
                '''
                actor_grad = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters  ( Update the frozen target models)
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
                ac_act_loss += actor_grad   # 221006

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        writer.add_scalar("(Critic) loss", av_loss / iterations, self.iter_count)
        writer.add_scalar("Actor loss", ac_act_loss / iterations, self.iter_count)  # 221006
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
#batch_size = 256  # 221007
batch_size = 200  # 221007
#discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)   # TODO 0.99로 바꿔보기
discount = 0.99   # 221007
tau = 0.005  # Soft target update variable (should be close to 0) 
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise 
policy_freq = 2  # Frequency of Actor network updates (delayed policy updates cycle)
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = True  # Weather to load a stored model   
random_near_obstacle = False  # To take random actions near obstacles or not
start_timesteps = 25e6 # 221006   # https://github.com/sfujim/TD3/blob/master/main.py 

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
network = TD3(state_dim, action_dim, max_action)
##221006
action_bound = [[0, -1], [1, 1]] 
action_bound = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
agent = SAC(num_frame_obs=1, num_goal_obs=2, num_vel_obs=2, action_space=action_bound)
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
                #evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)    
                evaluate(network=network, epoch=epoch, agent=agent, eval_episodes=eval_ep)   
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

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
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action)   
    
    '''
    if replay_buffer.count > 1024:
        for i in range(1):
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(replay_buffer, 512, updates)  # batch_size = 1024, updates = 1
            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1
    '''
    #action_sac = agent.select_action(np.array(state))
        #action = env.get_global_action()



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
    #next_state, reward, done, target = env.step(action_sac)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)
    #replay_buffer.add(state, action_sac, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_reward += reward
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, agent=agent, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
