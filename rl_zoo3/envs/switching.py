import gym
from gym import spaces

class SwitchingEnv(gym.Env):
  """Environment that switches between two environments at a given frequency"""

  def __init__(self, env1, env2, switch_freq):
    """
       :param switch_freq: number of timesteps before switching environments
    """
    super(SwitchingEnv, self).__init__()
    self.env1 = gym.make(env1)
    self.env2 = gym.make(env2)
    self.curr_env = self.env1
    self.switch_freq = switch_freq
    self.next_switch_step = self.switch_freq
    self.curr_step = 0

    self.action_space = self.curr_env.action_space
    self.observation_space = self.curr_env.observation_space

  def step(self, action):
    self.curr_step += 1
    return self.curr_env.step(action)

  def reset(self):
    if self.curr_step >= self.next_switch_step:
        self._switch_envs()
    
    return self.curr_env.reset()

  def render(self, mode='human'):
    return self.curr_env.render(mode)

  def close (self):
    self.env1.close()
    self.env2.close()

  def _switch_envs(self):
    self.next_switch_step += self.switch_freq
    self.curr_env = self.env2 if self.curr_env == self.env1 else self.env1
    self.action_space = self.curr_env.action_space
    self.observation_space = self.curr_env.observation_space
