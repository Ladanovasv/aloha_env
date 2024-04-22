from omni.isaac.gym.vec_env import VecEnvBase
from stable_baselines3 import PPO

env = VecEnvBase(headless=False)
from tasks.pick_place import AlohaTask
task = AlohaTask(name="Aloha", n_envs=1)
env.set_task(task, backend="torch")



# Choose the policy path to visualize
policy_path = "./standalone_examples/aloha-tdmpc/models/pick_place.zip"

model = PPO.load(policy_path)

for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = env.step(actions)
        env.render()

env.close()

