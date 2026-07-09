import os
import shutil
import hydra
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from crci_mem.envs.assistant_env import AdaptiveAssistanceEnv
from crci_mem.user_model.agent_forgetful import ForgetfulHH
from crci_mem.user_model.wrappers import CombinedExtractor_Belief
from crci_mem.inference.nested_particle_filter import AssistantParticleFilter
from typing import Callable, Optional, Dict, Union, List
import torch
import time
import wandb
from functools import partial

@hydra.main(version_base=None, config_path="../../configs", config_name="assistant_config.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    global best_models
    best_models = []
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")
    set_random_seed(cfg.utils.seed)
    np.random.seed(cfg.utils.seed)

    model_dir = os.path.join(cfg.dirpath.model, 'ai_assistant')
    log_dir = os.path.join(cfg.dirpath.log, 'ai_assistant')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = None
    if cfg.logging.wandb_enabled:
        run = wandb.init(project=cfg.logging.wandb_project, group="assistant",
                         name=f"assistant_{cfg.run_id}",
                         config=OmegaConf.to_container(cfg, resolve=True),
                         sync_tensorboard=True, reinit=True)

    policy_dir = os.path.join(cfg.dirpath.policy_model, 'user_models', cfg.env.name, f"model{cfg.agent.policy_model}")
    if not os.path.isdir(policy_dir):
        raise SystemExit(f"No Model {cfg.agent.policy_model} user policies at {policy_dir}."
                         f"first (curriculum_train.py for Model B), matching run_id/policy_run_id.")
    policy_files = [f for f in os.listdir(policy_dir) if f.endswith('.zip')]
    policy_files.sort()
    policy_files = [os.path.join(policy_dir, f) for f in policy_files]

    theta2policy_dir = {}
    for f in policy_files:
        try:
            theta = float(f.split('/')[-1].split('_')[1])
            theta2policy_dir[theta] = f
        except (ValueError, IndexError):
            print(f"Skipping non-per-theta policy file {f}")
    if not theta2policy_dir:
        raise SystemExit(f"No per-theta policies in {policy_dir}.")

    optimal_policy_path = None
    if hasattr(cfg.assistant, 'optimal_policy') and hasattr(cfg.assistant.optimal_policy, 'mdp_file'):
        mdp_filename = cfg.assistant.optimal_policy.mdp_file
        mdp_policy_dir = os.path.join(cfg.dirpath.policy_model, 'user_models', cfg.env.mdp_name)
        optimal_policy_path = os.path.join(mdp_policy_dir, mdp_filename)
        if not os.path.exists(optimal_policy_path):
            print(f"Warning: Optimal MDP policy not found at {optimal_policy_path}")
            optimal_policy_path = None
    
    if optimal_policy_path:
        print(f"Found optimal MDP policy at {optimal_policy_path}")

    simulated_agent_params = {
        'env': None,
        'model_cls': PPO,
        'save_dir': '',
        'seed': cfg.utils.seed,
        'device': 'cpu',  # inner particle-filter policies run on CPU: many tiny forward passes, avoids GPU contention
        'deterministic': cfg.agent.deterministic,
        'temperature': cfg.agent.temperature,
        'entropy_threshold': cfg.assistant.entropy_threshold,
        'memory_model': cfg.agent.memory_model,   # simulated user must match the true user (A/B)
    }
    
    crci_params = {
        'candidate_thetas': list(theta2policy_dir.keys()),
        'M': cfg.crci.M,
        'simulated_agent_class': ForgetfulHH,
        'simulated_agent_params': simulated_agent_params,
        'theta2policy_dir': theta2policy_dir,
        'seed': cfg.utils.seed,
        'model_class': PPO,
        'device': 'cpu',  # inner particle-filter policies run on CPU: many tiny forward passes, avoids GPU contention
        'temperature': cfg.agent.temperature,
    }

    env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode)
    env.reset(seed=cfg.utils.seed)
    assist_env = AdaptiveAssistanceEnv(
        base_env=env,
        theta2policy_dir=theta2policy_dir,
        crci_params=crci_params,
        assistance_cost={
            0: cfg.assistant.cost.no_assist,
            1: cfg.assistant.cost.action_hint,
            2: cfg.assistant.cost.memory_hint
        },
        model_class=PPO,
        device=cfg.utils.device,
        seed=cfg.utils.seed,
        optimal_policy_path=optimal_policy_path,
        episodes_per_agent=cfg.assistant.episodes_per_agent,
        entropy_threshold=cfg.assistant.entropy_threshold,
        temperature=cfg.agent.temperature,
        deterministic=cfg.agent.deterministic,
        memory_model=cfg.agent.memory_model,
    )

    model = PPO(
        policy=cfg.assistant.policy_type,
        env=assist_env,
        verbose=1,
        tensorboard_log=log_dir,
        device=cfg.utils.device,
    )
    
    for timestep in range(0, cfg.training.timesteps_per_iteration * cfg.training.n_iterations + 1, cfg.training.eval_freq):  
        model.learn(total_timesteps=cfg.training.eval_freq, reset_num_timesteps=False)
        mean_rw, _ = evaluate_policy(model, model.get_env(), cfg.training.n_eval_episodes)
        save_best_model(cfg.assistant.policy_type, model, mean_rw, timestep, model_dir, cfg.training.num_best_models)
        if run is not None:
            run.log({"eval/mean_reward": mean_rw}, step=timestep)
    model.save(os.path.join(model_dir, f'{cfg.assistant.policy_type}_{cfg.training.timesteps_per_iteration * cfg.training.n_iterations}'))
    if best_models:  
        shutil.copy(best_models[0][1], os.path.join(model_dir, f'{cfg.assistant.policy_type}_best.zip'))
    if run is not None:
        run.finish()

def save_best_model(name, model, reward, timestep, log_dir, num_best_models=5):
    global best_models
    model_path = os.path.join(log_dir, f"{name}_{timestep}.zip")
    model.save(model_path)

    best_models.append((reward, model_path))
    best_models = sorted(best_models, key=lambda x: -x[0]) 

    if len(best_models) > num_best_models:
        _, worst_model_path = best_models.pop()
        if os.path.exists(worst_model_path):
            os.remove(worst_model_path)

    print(f"Saved model with reward {reward:.2f} at {model_path}. Current best rewards: {[m[0] for m in best_models]}")

if __name__ == "__main__":
    main()