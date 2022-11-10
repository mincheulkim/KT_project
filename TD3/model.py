import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    
    
    
    
    
    


class QNetwork_PATH(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_PATH, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions+256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.traj1 = nn.Linear(10, hidden_dim)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions+256, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.traj2 = nn.Linear(10, hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        traj = state[:,-10:]
        #print('traj:',traj, traj.shape)
        state = state[:,:24]
        #print('state:',state)
        t1 = F.relu(self.traj1(traj))   # 256
        t2 = F.relu(self.traj2(traj))   # 256
        #state1 = torch.cat([state, t1], 1)
        #state2 = torch.cat([state, t2], 1)
        state1 = torch.cat((state, t1), dim=-1)
        state2 = torch.cat((state, t2), dim=-1)

        
        xu1 = torch.cat([state1, action], 1)
        xu2 = torch.cat([state2, action], 1)
        
        x1 = F.relu(self.linear1(xu1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy_PATH(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_PATH, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs+256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.traj = nn.Linear(10, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        #print('state:',state, state.shape)
        traj = state[:,-10:]
        #print('traj:',traj, traj.shape)
        state = state[:,:24]
        #print('state:',state)
        t = F.relu(self.traj(traj))   # 256
        #state = torch.cat([state, traj], 1)
        state = torch.cat((state, t), dim=-1)
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_PATH, self).to(device)







class QNetwork_PATH_ICRA2019(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_PATH_ICRA2019, self).__init__()

        # Q1 architecture
        self.goal1 = nn.Linear(2, hidden_dim)
        self.waypoint1 = nn.Linear(12, hidden_dim)
        self.fea_cv11 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=5, stride = 2, padding=1)
        self.fea_cv21 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fea_cv31 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.linear1 = nn.Linear(256+256+96+num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.traj1 = nn.Linear(10, hidden_dim)

        # Q2 architecture
        self.goal2 = nn.Linear(2, hidden_dim)
        self.waypoint2 = nn.Linear(12, hidden_dim)
        self.fea_cv12 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=5, stride = 2, padding=1)
        self.fea_cv22 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fea_cv32 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.linear4 = nn.Linear(256+256+96+num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.traj2 = nn.Linear(10, hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        #1. waypoint
        waypoint = state[:,-12:]    # waypoints  (6 x 2 = 12)
        #2. lidar
        lidar = state[:,:20]    # lidar 20
        lidar = lidar.unsqueeze(1)
        #3. goal
        goal = state[:, 20:22]  # goal 2
        
        # Q1
        w1 = F.relu(self.waypoint1(waypoint))
        g1 = F.relu(self.goal1(goal))
        l1 = F.relu(self.fea_cv11(lidar))
        l1 = F.relu(self.fea_cv21(l1))
        l1 = F.relu(self.fea_cv31(l1))
        l1 = l1.view(l1.shape[0], -1)
        state1 = torch.cat((w1, g1, l1), dim=-1)
        
        
        xu1 = torch.cat([state1, action], 1)
        

        x1 = F.relu(self.linear1(xu1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2
        w2 = F.relu(self.waypoint2(waypoint))
        g2 = F.relu(self.goal2(goal))
        l2 = F.relu(self.fea_cv12(lidar))
        l2 = F.relu(self.fea_cv22(l2))
        l2 = F.relu(self.fea_cv32(l2))
        l2 = l2.view(l2.shape[0], -1)
        state2 = torch.cat((w2, g2, l2), dim=-1)
        
        xu2 = torch.cat([state2, action], 1)
        
        x2 = F.relu(self.linear4(xu2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy_PATH_ICRA2019(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_PATH_ICRA2019, self).__init__()
        # Goal: FC256
        self.goal = nn.Linear(2, hidden_dim)
        
        # Waypoints: FC256
        self.waypoint = nn.Linear(12, hidden_dim)

        # Lidar: 1DCNN 1DCNN 1DCNN FC256
        self.fea_cv1 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=5, stride = 2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fea_cv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.linear1 = nn.Linear(256+256+96, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):   # state 0~19: lidar(20), 20~23: robot state(4), 24~35: waypoints six(12)
        #1. waypoint
        waypoint = state[:,-12:]    # waypoints  (6 x 2 = 12)
        #2. lidar
        lidar = state[:,:20]    # lidar 20
        lidar = lidar.unsqueeze(1)

        #3. goal
        goal = state[:, 20:22]  # goal 2
        
        w = F.relu(self.waypoint(waypoint))
        g = F.relu(self.goal(goal))
        l = F.relu(self.fea_cv1(lidar))   # batch, channel, feature
        l = F.relu(self.fea_cv2(l))
        l = F.relu(self.fea_cv3(l))
        l = l.view(l.shape[0], -1)
        state = torch.cat((w, g, l), dim=-1)   # w 256, g 256, l 96, state = 608
                
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_PATH_ICRA2019, self).to(device)