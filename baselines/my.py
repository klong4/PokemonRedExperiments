from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from argparse_pokemon import *

# Common Configuration
sess_path = f'session_{str(uuid.uuid4())[:8]}'
run_steps = 2048 * 8
runs_per_update = 6
updates_per_checkpoint = 4

args = get_args('run_baseline.py', ep_length=run_steps, sess_path=sess_path)

env_config = {
    'headless': False, 'save_final_state': True, 'early_stop': False,
    'action_freq': 24, 'init_state': '../fast_text_start.state', 'max_steps': run_steps,
    'print_rewards': True, 'save_video': True, 'session_path': sess_path,
    'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
}

env_config = change_env(env_config, args)
env = RedGymEnv(config=env_config)

env_checker.check_env(env)

learn_steps = 40
file_name = 'poke_'  # Modify this if needed
inference_only = True

# Training Part
if not inference_only:
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "n_steps": 10 * 20 * 2048
        }
        model = PPO.load(file_name, env=env, custom_objects=custom_objects)
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=run_steps * runs_per_update, batch_size=512, n_epochs=10,
                    gamma=0.98)

    for i in range(learn_steps):
        model.learn(total_timesteps=run_steps * runs_per_update * updates_per_checkpoint)
        model.save(sess_path / Path(file_name + str(i)))

# Running Part
else:
    def make_env(rank, env_conf, seed=0):
        def _init():
            env = RedGymEnv(env_conf)
            env.reset(seed=(seed + rank))
            return env
        set_random_seed(seed)
        return _init

    ep_length = 2 ** 16
    args = get_args(usage_string=None, headless=False, ep_length=ep_length, sess_path=sess_path)

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
    }
    env_config = change_env(env_config, args)

    num_cpu = 6  # Change this to the desired number of CPU processes
    env = make_env(0, env_config)()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    file_name = 'session_4da05e87_main_good/poke_439746560_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()

    obs, info = env.reset()
    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
