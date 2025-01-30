import yaml
import itertools

from simulator import simulation, conv_simulation, atari_simulation
from agent import Agent
from collector import Collector
from utils.get_screen_utils import get_screen  # TODO: frame_shape廃止時に削除
from utils.space2size_utils import space2size
from utils.create_param_folder_utils import create_param_folder
from utils.calculate_executed_sim_utils import calculate_executed_sim
from utils.create_env_utils import create_env
from utils.algo_class_utils import ALGO_CLASS
from utils.default_param_utils import apply_default_params


if __name__ == '__main__':
    main_yaml_path = 'config/main.yaml'
    with open(main_yaml_path, 'r') as f:
        main_config = yaml.safe_load(f)

    expt_yaml_files = main_config['experiments']

    for expt_yaml_file in expt_yaml_files:
        with open(expt_yaml_file, 'r') as f:
            expt_config = yaml.safe_load(f)

        expt_config['param'] = apply_default_params(expt_config['param'])
        algo_list = expt_config['algo']
        param_keys = expt_config['param'].keys()
        param_values = expt_config['param'].values()
        param_combinations = list(itertools.product(*param_values))

        params = []
        for algo in algo_list:
            for param_combination in param_combinations:
                param_dict = dict(zip(param_keys, param_combination))
                param_dict.update({
                    'algo': algo,
                    'env': expt_config['env'],
                    'sim': expt_config['sim'],
                    'epi': expt_config['epi']
                })
                params.append(param_dict)
        print(f'Number of experiments: {len(params)}')

        for param in params:
            env = create_env(param['env'], param['algo'])
            env.reset()
            init_frame = get_screen(env)  # TODO: frame_shape廃止時に削除

            model_class, policy_class = ALGO_CLASS.get(param['algo'])

            param.update({
                'action_space': space2size(env.action_space),
                'state_space': space2size(env.observation_space),
                'frame_shape': init_frame.shape,  # TODO: frame_shape廃止時に削除
                'model_class': model_class
            })

            result_dir_path = create_param_folder(param['env'], param['algo'], param)
            executed_sims = calculate_executed_sim(result_dir_path)
            policy = policy_class(**param)
            agent = Agent(policy)
            collector = Collector(param['sim'], param['epi'], param, agent, policy, result_dir_path)

            with open(result_dir_path + 'config.yaml', 'w') as f:
                yaml.dump(expt_config, f, allow_unicode=True, default_flow_style=False)

            # TODO: Atariアルゴリズムを採用，Convを削除，のちにAtari→Convに変更
            if 'Conv' in param['algo']:
                if 'Atari' in param['algo']:
                    atari_simulation(param['sim'], param['epi'], agent, collector, result_dir_path)
                else:
                    conv_simulation(param['sim'], param['epi'], env, agent, collector, param['neighbor_frames'], result_dir_path)
            else:
                simulation(param['sim'], executed_sims, param['epi'], env, agent, collector, result_dir_path)
