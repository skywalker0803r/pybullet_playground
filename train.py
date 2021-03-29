import pybullet_envs.bullet.minitaur_gym_env as e
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

total_timesteps = 1000000
env = e.MinitaurBulletEnv(render=False)
model = PPO2(MlpPolicy,env=env,verbose=1,tensorboard_log="./tensorboard")
model.learn(total_timesteps=total_timesteps)
model.save("./model/model{}".format(total_timesteps))
print("model training done!")
