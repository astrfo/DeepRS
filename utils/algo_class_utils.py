from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.ps import PS
from policy.rs2_ambition_em_dqn import RS2AmbitionEMDQN
from policy.rs2_ambition_dqn import RS2AmbitionDQN
from policy.rs2_dqn import RS2DQN
from policy.conv_dqn import ConvDQN
from policy.conv_rs2_ambition_em_dqn import ConvRS2AmbitionEMDQN
from policy.conv_rs2_ambition_dqn import ConvRS2AmbitionDQN

from network.qnet import QNet
from network.rs2emnet import RS2EMNet
from network.rs2net import RS2Net
from network.conv_qnet import ConvQNet
from network.conv_rs2emnet import ConvRS2EMNet
from network.conv_rs2net import ConvRS2Net


ALGO_CLASS = {
    'DQN': (QNet, DQN),
    'DDQN': (QNet, DDQN),
    'PS': (QNet, PS),
    'RS2AmbitionEMDQN': (RS2EMNet, RS2AmbitionEMDQN),
    'RS2AmbitionDQN': (RS2Net, RS2AmbitionDQN),
    'RS2DQN': (RS2Net, RS2DQN),
    'ConvDQN': (ConvQNet, ConvDQN),
    'ConvRS2AmbitionEMDQN': (ConvRS2EMNet, ConvRS2AmbitionEMDQN),
    'ConvRS2AmbitionDQN': (ConvRS2Net, ConvRS2AmbitionDQN)
}
