import pybullet_envs.bullet.minitaur_gym_env as e
from DDPG import DDPGAgent

env = e.MinitaurBulletEnv(render=True)
agent = DDPGAgent(env)
agent.train(max_episodes=1000,max_steps=100000,batch_size=64)
