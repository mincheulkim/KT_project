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

        '''
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
        '''
        # Q1 architecture
        self.goal1 = nn.Linear(2, hidden_dim)
        self.robot_s1 = nn.Linear(4, hidden_dim)   # 230206
        self.waypoint1 = nn.Linear(10, hidden_dim)
        self.fea_cv11 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=5, stride = 2, padding=1)
        self.fea_cv21 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fea_cv31 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.linear1 = nn.Linear(256+256+96+num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.traj1 = nn.Linear(10, hidden_dim)

        # Q2 architecture
        self.goal2 = nn.Linear(2, hidden_dim)
        self.robot_s2 = nn.Linear(4, hidden_dim)  # 230206
        self.waypoint2 = nn.Linear(10, hidden_dim)
        self.fea_cv12 = nn.Conv1d(in_channels = 1, out_channels=32, kernel_size=5, stride = 2, padding=1)
        self.fea_cv22 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fea_cv32 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.linear4 = nn.Linear(256+256+96+num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.traj2 = nn.Linear(10, hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        '''
        traj = state[:,-10:]
        state = state[:,:24]
        t1 = F.relu(self.traj1(traj))   # 256
        t2 = F.relu(self.traj2(traj))   # 256
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
        '''
        # state: 0~19(20) : lidar, 20~23(robot state as g_x, g_y, v, w), 24~33(10): waypoints
        
        #1. waypoint
        waypoint = state[:,-10:]    # waypoints  (5 x 2 = 10)
        #2. lidar
        lidar = state[:,:20]    # lidar 20
        lidar = lidar.unsqueeze(1)
        #3. goal
        #goal = state[:, 20:22]  # goal 2
        robot_state = state[:, 20:24]  # g_x, g_y, v, w
        
        # Q1
        w1 = F.relu(self.waypoint1(waypoint))
        #g1 = F.relu(self.goal1(goal))
        g1 = F.relu(self.robot_s1(robot_state))
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
        #g2 = F.relu(self.goal2(goal))
        g2 = F.relu(self.robot_s2(robot_state))
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


class GaussianPolicy_PATH(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_PATH, self).__init__()
        '''
        self.linear1 = nn.Linear(num_inputs+256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.traj = nn.Linear(10, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        '''
        # Goal: FC256
        self.goal = nn.Linear(2, hidden_dim)
        # Robot state: FC256
        self.robot_state = nn.Linear(4, hidden_dim)
        
        # Waypoints: FC256
        self.waypoint = nn.Linear(10, hidden_dim)

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

    def forward(self, state):
        '''
        traj = state[:,-10:]
        state = state[:,:24]
        t = F.relu(self.traj(traj))   # 256
        state = torch.cat((state, t), dim=-1)
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        '''
        #1. waypoint
        waypoint = state[:,-10:]    # waypoints  (5 x 2 = 10)
        #2. lidar
        lidar = state[:,:20]    # lidar 20
        lidar = lidar.unsqueeze(1)

        #3. goal
        goal = state[:, 20:22]  # goal 2
        robot_state = state[:, 20:24]
        #print('state:',state)
        #print('l:',lidar)
        #print('waypoint:',waypoint )
        #print('goal:',goal)
        
        w = F.relu(self.waypoint(waypoint))
        #g = F.relu(self.goal(goal))
        g = F.relu(self.robot_state(robot_state))
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


##### Library for DRLVO #####
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2 #4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    
    


####### 240219 ######################

class QNetwork_DRLVO(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_DRLVO, self).__init__()
        block = Bottleneck
        layers = [2, 1, 1]
        zero_init_residual=True
        groups=1
        width_per_group=64
        replace_stride_with_dilation=None
        norm_layer=None

        ################## ped_pos net model: ###################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#### For Q2
        self.conv1_2 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1_2 = norm_layer(self.inplanes)
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1_2 = self._make_layer(block, 64, layers[0])
        self.layer2_2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.conv2_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv3_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3_2 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)                


        self.linear1 = nn.Linear(512+2+2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(512+2+2, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)        
        

        self.apply(weights_init_)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, state, action):
        '''
        traj = state[:,-10:]
        state = state[:,:24]
        t1 = F.relu(self.traj1(traj))   # 256
        t2 = F.relu(self.traj2(traj))   # 256
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
        '''
        #### 0. state 정렬
        ped_pos = state[:, :12800]   # 12800
        scan = state[:, 12800:19200] # 6400
        goal = state[:, 19200:]      # 2
        ###### 1.Start of fusion net ######
        ped_in = ped_pos.reshape(-1,2,80,80)
        scan_in = scan.reshape(-1,1,80,80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)
                
        

        # Q1
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)

        identity2 = self.downsample2(x)

        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)


        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)
        ###### End of fusion net ######
        
        ###### 2.Start of goal net #######
        goal_in = goal.reshape(-1,2)
        goal_out = torch.flatten(goal_in, 1)
        ###### End of goal net #######        
        
        # 3. Combine
        state1 = torch.cat((fusion_out, goal_out), dim=-1)    # 512 + 2 = 514
        
        xu1 = torch.cat([state1, action], 1)

        x1 = F.relu(self.linear1(xu1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2
        x_2 = self.conv1_2(fusion_in)
        x_2 = self.bn1_2(x_2)
        x_2 = self.relu_2(x_2)
        x_2 = self.maxpool_2(x_2)

        identity3_2 = self.downsample3_2(x_2)

        x_2 = self.layer1_2(x_2)

        identity2_2 = self.downsample2_2(x_2)

        x_2 = self.layer2_2(x_2)

        x_2 = self.conv2_2_2(x_2)
        x_2 += identity2_2
        x_2 = self.relu2_2(x_2)


        x_2 = self.layer3(x_2)

        x_2 = self.conv3_2_2(x_2)
        x_2 += identity3_2
        x_2 = self.relu3_2(x_2)

        x_2 = self.avgpool_2(x_2)
        fusion_out_2 = torch.flatten(x_2, 1)
        ###### End of fusion net ######
        
        ###### 2.Start of goal net #######
        goal_in = goal.reshape(-1,2)
        goal_out = torch.flatten(goal_in, 1)
        ###### End of goal net #######        
        
        # 3. Combine
        state2 = torch.cat((fusion_out_2, goal_out), dim=-1)    # 512 + 2 = 514
        
        xu2 = torch.cat([state2, action], 1)

        x2 = F.relu(self.linear4(xu2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)


        print('x1:',x1, 'x2:',x2)
        return x1, x2


class GaussianPolicy_DRLVO(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_DRLVO, self).__init__()
        '''
        self.linear1 = nn.Linear(num_inputs+256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.traj = nn.Linear(10, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        '''
        # network parameters:
        block = Bottleneck
        layers = [2, 1, 1]
        zero_init_residual=True
        groups=1
        width_per_group=64
        replace_stride_with_dilation=None
        norm_layer=None

        # inherit the superclass properties/methods
        #
        # define the model
        #
        ################## ped_pos net model: ###################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)                


        self.linear1 = nn.Linear(512+2, hidden_dim)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, state):
        #### 0. state 정렬
        ped_pos = state[:, :12800]   # 12800
        scan = state[:, 12800:19200] # 6400
        goal = state[:, 19200:]      # 2
        
        ###### 1.Start of fusion net ######
        ped_in = ped_pos.reshape(-1,2,80,80)
        scan_in = scan.reshape(-1,1,80,80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)
        
        # See note [TorchScript super()]
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)

        identity2 = self.downsample2(x)

        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)


        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)
        ###### End of fusion net ######
        
        ###### 2.Start of goal net #######
        goal_in = goal.reshape(-1,2)
        goal_out = torch.flatten(goal_in, 1)
        ###### End of goal net #######        
        
        # 3. Combine
        state = torch.cat((fusion_out, goal_out), dim=-1)    # 512 + 2 = 514

        # 4. Original code                
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
        return super(GaussianPolicy_DRLVO, self).to(device)