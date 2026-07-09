import os
import pickle
from copy import deepcopy
import pandas as pd
import gymnasium as gym
from crci_mem.pipeline.base import BasePipeline
from crci_mem.pipeline import plotting

"""
In this pipeline, we use the forgetful agent model (rl agent) to test the nested particle filter method.
"""

class InferencePipeline(BasePipeline):
    def __init__(self, env: gym.Env=None, theta_true: float=0.0, agent_params: dict=None, agent_class=None, crci_class=None, crci_params: dict=None, save_dir: str='', seed: int=42):
        super().__init__(theta_true, agent_params, agent_class, crci_class, crci_params)
        self.trajs = []
        self.save_dir = save_dir
        if env is not None:
            self.env = env.unwrapped

    def reset_streaming(self):
        self.env.reset()
        self.agent.reset(env=self.env, load_policy=False)
        self.crci.reset_streaming(env=self.env)


    def online_theta_posterior_estimation_streaming(self, T: int=100):
        '''
        This function is used to simulate the online theta posterior estimation process during interaction with the agent up to step T.
        '''

        step = 0
        while step < T:
            self.reset_streaming()
            deepcopy_env = deepcopy(self.env)
            traj = []
            done = False
            while not done and step < T:
                s = self.env._get_state()
                o = self.env._get_obs()
                m, a, b, reward, done = self.agent.step()
                traj.append({
                    's': s,
                    'o': o,
                    'm': m,
                    'a': a,
                    'b': b,
                    'reward': reward,
                })
                if step == 0:
                    self.crci.initialize(s, a, streaming=False)
                elif len(traj) == 1:
                    self.crci.initialize(s, a, streaming=True)
                else:
                    self.crci.update(s, a)
                step += 1
            traj.append({
                    's': self.env._get_state(),
                    'o': self.env._get_obs(),
                    'm': None,
                    'a': None,
                    'b': None,
                    'reward': reward,
                })
            self.trajs.append({
                'env': deepcopy_env,
                'trajectory': traj,
            })
        self.posterior = self.crci.get_posterior()
        return self.posterior, self.trajs[0]

    def plot_trajectory(self, df=None, theta=None, env=None, traj=None, filename=None):
        return plotting.plot_trajectory(df=df, theta=theta, env=env, traj=traj, filename=filename, save_dir=self.save_dir)

    def plot_animated_trajectory(self, theta=None, env=None, traj=None, filename=None):
        return plotting.plot_animated_trajectory(theta=theta, env=env, traj=traj, filename=filename, save_dir=self.save_dir)

    def data_processing_streaming(self, data_dir: str, save_dir: str):
        try:
            results_file = [f for f in os.listdir(data_dir) if f.endswith('pkl')]
            results_file = [os.path.join(data_dir, f) for f in results_file]
            results = []
            for file in results_file:
                temp = pickle.load(open(file, 'rb'))
                results.append(temp)
            results = pd.DataFrame(results)
            results = results.sort_values(by=['theta_true']).reset_index(drop=True)
            refactored_data_df = pd.DataFrame(columns=['step','theta_true','seed', 'temp', 'map', 'map_error', 'posterior_mean', 'mean_error', 'posterior'])
            for i in range(len(results)):
                entry = results.iloc[i]
                data = entry['data']
                #traj = entry['env_trajectory']
                for j in range(len(data)):
                    refactored_data_df = pd.concat([
                        refactored_data_df,
                        pd.DataFrame([{
                            'step': data[j]['step'],
                            'theta_true': entry['theta_true'],
                            'seed': entry['seed'],
                            'temp': entry['temperature'],
                            'map': data[j]['map'],
                            'map_error': data[j]['map_error'],
                            'posterior_mean': data[j]['posterior_mean'],
                            'mean_error': data[j]['mean_error'],
                            'posterior': data[j]['posterior'],
                        }])
                    ], ignore_index=True)
            results = results.drop(columns=['data'])
            save_name = os.path.join(save_dir, 'data.pkl')
            refactored_data_df.to_pickle(save_name)
            env_traj_save_name = os.path.join(save_dir, 'env_traj.pkl')
            results.to_pickle(env_traj_save_name)
            print(f"Data processing completed. Refactored data saved to {save_name}, env traj saved to {env_traj_save_name}.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def plot_error_paper_2(self, df_raw, temp=None):
        return plotting.plot_error_paper_2(df_raw=df_raw, temp=temp, save_dir=self.save_dir)

    def plot_static_posterior(self, theta_particles_history=None, theta_true=None, temp=None):
        return plotting.plot_static_posterior(theta_particles_history=theta_particles_history,
                                              theta_true=theta_true, temp=temp, save_dir=self.save_dir)

    def plot_animated_theta_posterior(self, theta_particles_history=None, theta_true=None, temp=None):
        return plotting.plot_animated_theta_posterior(theta_particles_history=theta_particles_history,
                                                      theta_true=theta_true, temp=temp, save_dir=self.save_dir)
