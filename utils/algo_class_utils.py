from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.ps import PS
from policy.rsrsaleph_q_eps_ras_choice_dqn import RSRSAlephQEpsRASChoiceDQN
from policy.rs2_ambition_dqn import RS2AmbitionDQN
from policy.rs2_dqn import RS2DQN
from policy.conv_dqn_atari import ConvDQNAtari
from policy.conv_rsrsaleph_q_eps_ras_choice_dqn_atari import ConvRSRSAlephQEpsRASChoiceDQNAtari
from policy.conv_rs2_ambition_dqn import ConvRS2AmbitionDQN

from network.qnet import QNet
from network.rsrsnet import RSRSNet
from network.rsrsdqnnet import RSRSDQNNet
from network.conv_atari_qnet import ConvQAtariNet
from network.conv_atari_rsrsnet import ConvRSRSAtariNet


ALGO_CLASS = {
    'DQN': (QNet, DQN),
    'DDQN': (QNet, DDQN),
    'PS': (QNet, PS),
    'RSRSAlephQEpsRASChoiceDQN': (RSRSNet, RSRSAlephQEpsRASChoiceDQN),
    'RS2AmbitionDQN': (RSRSDQNNet, RS2AmbitionDQN),
    'RS2DQN': (RSRSDQNNet, RS2DQN),
    'ConvDQNAtari': (ConvQAtariNet, ConvDQNAtari),
    'ConvRSRSAlephQEpsRASChoiceDQNAtari': (ConvRSRSAtariNet, ConvRSRSAlephQEpsRASChoiceDQNAtari),
    'ConvRS2AmbitionDQN': (ConvRSRSAtariNet, ConvRS2AmbitionDQN)
}
