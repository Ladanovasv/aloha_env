import torch as th

from omni.isaac.gym.vec_env import VecEnvBase
from stable_baselines3 import PPO

env = VecEnvBase(headless=True)
from tasks.reach_target import AlohaTask
task = AlohaTask(name="Aloha", n_envs=1)
env.set_task(task, backend="torch")


policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])]) # Policy params

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    device="cuda",
    gae_lambda=1.0,
    vf_coef=0.95,
    max_grad_norm=10,
    tensorboard_log="./standalone_examples/aloha-tdmpc/logs/"
)
model.learn(total_timesteps=50000)
model.save("./standalone_examples/aloha-tdmpc/models/reach_target")

env.close()