from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np


env = DummyVecEnv(
    [lambda: Monitor(TimeLimit(HIVPatient(domain_randomization=False,
                                              logscale=False),
                               max_episode_steps=200)),
     lambda: Monitor(TimeLimit(HIVPatient(domain_randomization=True,
                                              logscale=False),
                               max_episode_steps=200))
                               ]
)
env = VecNormalize(env, norm_reward=True, norm_obs=True)

# The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ProjectAgent:
    def __init__(self, trained=True, batch_size=40):
        self.env = env
        self.trained = trained
        self.model = PPO("MlpPolicy",
                         env,
                         n_steps=200,
                         batch_size=batch_size,
                         gae_lambda=0.9,
                         clip_range=0.25,
                         n_epochs=10,
                         )

        if not trained:
            self.train()

    def act(self, observation):
        if self.trained:
            # Normalized observation
            observation = (observation - self.mean) / self.std
        action, _ = self.model.predict(observation)
        return action

    def train(self, n_steps=200, max_episodes=10000):

        self.model = self.model.learn(total_timesteps=max_episodes*n_steps,
                                      progress_bar=True)
        self.trained = True
        self.save()

    def load(self, path="../vecnormalize_stats2.pkl"):
        # load env params
        path_env = 'mva-rl-assignment-chris-mrn/vecnormalize_stats2.pkl'
        path_model = 'mva-rl-assignment-chris-mrn/ppo_model.zip'
        env = VecNormalize.load(path_env, self.env)
        # if there is a model, load it

        self.model = PPO.load(path_model)
        print("Model loaded!")

        self.mean = env.obs_rms.mean
        self.std = np.sqrt(env.obs_rms.var)
        self.trained = True

    def save(self, path='mva-rl-assignment-chris-mrn/vecnormalize_stats2.pkl'):
        # save env params
        self.env.save(path)
        print("env saved")
        # save model parms
        self.model.save("ppo_model")
        print("Model saved!")