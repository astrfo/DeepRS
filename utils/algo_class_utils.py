from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.ps import PS
from policy.rsrsaleph_q_eps_ras_choice_dqn import RSRSAlephQEpsRASChoiceDQN
from policy.rsrsaleph_q_eps_ras_choice_centroid_dqn import RSRSAlephQEpsRASChoiceCentroidDQN
from policy.rsrsaleph_q_eps_ras_choice_centroid_grcw_dqn import RSRSAlephQEpsRASChoiceCentroidGRCwDQN
from policy.conv_dqn_atari import ConvDQNAtari
from policy.conv_rsrsaleph_q_eps_ras_choice_dqn_atari import ConvRSRSAlephQEpsRASChoiceDQNAtari
from policy.conv_rsrsaleph_q_eps_ras_choice_centroid_dqn_atari import ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari

from network.qnet import QNet
from network.rsrsnet import RSRSNet
from network.rsrsdqnnet import RSRSDQNNet
from network.conv_qnet import ConvQNet
from network.conv_atari_qnet import ConvQAtariNet
from network.conv_rsrsnet import ConvRSRSNet
from network.conv_rsrsalephnet import ConvRSRSAlephNet
from network.conv_atari_rsrsnet import ConvRSRSAtariNet


ALGO_CLASS = {
    'DQN': (QNet, DQN),
    'DDQN': (QNet, DDQN),
    'PS': (QNet, PS),
    'RSRSAlephQEpsRASChoiceDQN': (RSRSNet, RSRSAlephQEpsRASChoiceDQN),
    'RSRSAlephQEpsRASChoiceCentroidDQN': (RSRSDQNNet, RSRSAlephQEpsRASChoiceCentroidDQN),
    'RSRSAlephQEpsRASChoiceCentroidGRCwDQN': (RSRSDQNNet, RSRSAlephQEpsRASChoiceCentroidGRCwDQN),
    'ConvDQNAtari': (ConvQAtariNet, ConvDQNAtari),
    'ConvRSRSAlephQEpsRASChoiceDQNAtari': (ConvRSRSAtariNet, ConvRSRSAlephQEpsRASChoiceDQNAtari),
    'ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari': (ConvRSRSAtariNet, ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari)
}
