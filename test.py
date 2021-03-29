import pybullet_envs.bullet.minitaur_gym_env as e
from stable_baselines import PPO2

total_timesteps = 1000000
model = PPO2.load("./model/model{}".format(total_timesteps))
env = e.MinitaurBulletEnv(render=True)
obs = env.reset()
for i in range(50000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode="human")
env.close()
