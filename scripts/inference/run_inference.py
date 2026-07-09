import os
import pickle
import hydra
import numpy as np
import logging
import concurrent.futures
import multiprocessing as mp
from copy import deepcopy
import glob
from omegaconf import DictConfig
from stable_baselines3 import PPO
import torch
import gymnasium as gym
import wandb
from crci_mem.inference.nested_particle_filter import NestedParticleFilter
from crci_mem.user_model.agent_forgetful import ForgetfulHH, ForgetfulHH_no_reaction
from crci_mem.pipeline.pipeline import InferencePipeline  

def run_experiment(exp_id, cfg, theta_true, seed, temp, save_dir):
    """
    Run one experiment with the given parameters.
    
    Parameters:
        theta_true (float): The true parameter value for the agent.
        beta (float): The beta parameter in the agent's dynamics.
        M (int): The number of inner particles (M) for the nested particle filter.
        seed (int): Random seed for reproducibility.
        save_dir (str): Directory where any plots or logs may be saved.
    
    Returns:
        dict: A dictionary containing the experiment parameters, final results, and convergence evolution (list of records per timestep).
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Experiment {exp_id} running on {device}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    run = None
    if cfg.logging.wandb_enabled:
        run = wandb.init(project=cfg.logging.wandb_project, group="inference",
                         name=f"theta{theta_true}_seed{seed}_temp{temp}",
                         config={"theta_true": theta_true, "seed": seed, "temperature": temp}, reinit=True)

    env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode)
    env.reset(seed=seed)
    policy_dir = os.path.join(cfg.dirpath.policy_model, 'user_models', cfg.env.name, f"model{cfg.user_model.memory.model}")
    if not os.path.isdir(policy_dir):
        raise SystemExit(f"No Model {cfg.user_model.memory.model} policies at {policy_dir} -- "
                         f"train the user policies first (scripts/user_model/train.py), same run_id.")
    policy_files = [f for f in os.listdir(policy_dir) if f.endswith('.zip')]
    policy_files.sort()
    policy_files = [os.path.join(policy_dir, f) for f in policy_files]
    theta2policy_dir = {}
    for f in policy_files:
        if 'mdp' in f:
            continue
        theta = float(f.split('/')[-1].split('_')[1])
        theta2policy_dir[theta] = f
    if theta_true not in theta2policy_dir:
        raise SystemExit(f"No policy for theta={theta_true} in {policy_dir} "
                         f"(found {sorted(theta2policy_dir)}). Train it first.")

    agent_params = {
        'env': env,
        'model_cls': PPO,
        'theta': theta_true,
        'save_dir': save_dir,
        'true_policy_dir': theta2policy_dir[theta_true],
        'seed': seed,
        'device': cfg.utils.device,
        'deterministic': cfg.user_model.memory.deterministic,
        'temperature': temp, 
        'memory_model': cfg.user_model.memory.model,
    }

    simulated_agent_params = {
        'env': env,
        'model_cls': PPO,
        'save_dir': save_dir,
        'true_policy_dir': None,
        'seed': seed,
        'device': cfg.utils.device,
        'deterministic': cfg.crci.deterministic,
        'memory_model': cfg.user_model.memory.model,
    }

    crci_params = {
        'candidate_thetas': list(theta2policy_dir.keys()),
        'M': cfg.crci.M,
        'simulated_agent_class': ForgetfulHH_no_reaction,
        'simulated_agent_params': simulated_agent_params,
        'theta2policy_dir': theta2policy_dir,
        'seed': seed,
        'model_class': PPO,
        'device': cfg.utils.device,
        'temperature': temp,
        'adaptive': False,
    }

    pipe = InferencePipeline(
        env=env,
        theta_true=theta_true,
        agent_params=agent_params,
        agent_class=ForgetfulHH_no_reaction,
        crci_class=NestedParticleFilter,
        crci_params=crci_params,
        save_dir=save_dir,
        seed=seed
    )

    pipe.online_theta_posterior_estimation_streaming()

    data = []
    for step_idx, record in pipe.crci.theta_particles_history.items():
        posterior = record['posterior']
        map = max(posterior, key=posterior.get)
        map_error = abs(map - theta_true)
        theta_vals = list(posterior.keys())
        weights = list(posterior.values())
        posterior_mean = np.average(theta_vals, weights=weights)
        mean_error = abs(posterior_mean - theta_true)
        
        data.append({
            'step': step_idx,
            'posterior': posterior,
            'map': map,
            'map_error': map_error,
            'posterior_mean': posterior_mean,
            'mean_error': mean_error
        })
    
    if run is not None:
        for rec in data:
            run.log({"inference/mean_error": rec['mean_error'], "inference/map_error": rec['map_error']}, step=rec['step'])
        run.summary["final_mean_error"] = data[-1]['mean_error']
        run.finish()

    result = {
        'theta_true': theta_true,
        'seed': seed,
        'temperature': temp,
        'data': data,
        'env_trajectory': pipe.trajs,  
    }
    
    logging.info(f"Completed experiment: theta_true={theta_true}, seed={seed}, temp={temp}, final_mean_error={data[-1]['mean_error']}, final_map_error={data[-1]['map_error']}")
    return result

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # run_id (in dirpath.root) separates runs; all array tasks share this one dir. The model<A|B>
    # subdir keeps inference of a Model A vs Model B user apart within one run.
    mtag = f"model{cfg.user_model.memory.model}"
    result_dir = os.path.join(cfg.dirpath.result, 'inference', mtag)   
    raw_dir = os.path.join(result_dir, 'raw')
    log_dir = os.path.join(cfg.dirpath.log, 'inference', mtag)         
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "run.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
                        force=True)
    
    logging.info(f"Experiment parameters: {cfg}")

    theta_trues = [round(i * 0.1, 1) for i in range(11)]        
    seeds = [42, 3407, 1131, 2543, 66]                   
    #temps = [1.0, 5.0, 10.0]
    temps = [3.0]
    result_files = glob.glob(os.path.join(raw_dir, 'result_*.pkl'))
    completed_indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in result_files]

    experiments = []
    for i, theta_true in enumerate(theta_trues):
        for j, seed in enumerate(seeds):
            for k, temp in enumerate(temps):
                experiments.append((i*len(seeds)+j, cfg, theta_true, seed, temp, result_dir))

    logging.info(f"Total experiments to run: {len(experiments)}")

    exp_index_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if exp_index_env is not None:
        exp_index = int(exp_index_env)
        if exp_index in completed_indices:
            logging.info(f"Experiment {exp_index} already completed. Skipping...")
            return
        if exp_index < len(experiments):
            exp = experiments[exp_index]
            logging.info(f"Running experiment with index: {exp_index}")
            result = run_experiment(*exp)
            data_file = os.path.join(raw_dir, f"result_{exp_index}.pkl")
            with open(data_file, "wb") as f:
                pickle.dump(result, f)
            logging.info(f"Experiment {exp_index} completed. Result saved to {data_file}.")
        else:
            logging.error(f"Experiment index {exp_index} out of range. Total experiments: {len(experiments)}")
    else:
        logging.info("No SLURM_ARRAY_TASK_ID found. Running all experiments sequentially.")
        for exp in experiments:
            exp_index = exp[0]
            if exp_index in completed_indices:
                continue
            try:
                result = run_experiment(*exp)
                data_file = os.path.join(raw_dir, f"result_{exp_index}.pkl")
                with open(data_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as exc:
                logging.error(f"Experiment {exp} generated an exception: {exc}")

if __name__ == "__main__":
    main()
