"""
Train per-theta ForgetfulHH user policies with PPO.

The memory model is selected by `user_model.memory.model` ('A' resample-fresh, default; 'B'
persistent). Train a single theta via `user_model.memory.decay_rate`, or several in one run by
passing the `user_model.memory.thetas` list.

The list trains sequentially in one job; under a SLURM array (--array=0-10) each task trains ONE
theta of the list, in parallel across the cluster. 
"""
import os
import shutil
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

import crci_mem  
from crci_mem.user_model.agent_forgetful import make_belief_wrapper
from crci_mem.user_model.wrappers import CombinedExtractor_Obs, CombinedExtractor_Belief

ALGOS = {"ppo": PPO, "dqn": DQN}
EXTRACTORS = {"obs": CombinedExtractor_Obs, "belief": CombinedExtractor_Belief}


def make_env(cfg, theta):
    env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode)
    if not cfg.env.use_belief:
        return env
    return make_belief_wrapper(env, theta, cfg.user_model.memory.model)


def train_one(cfg, theta):
    """Train and save a single policy for cognitive bound `theta` (the MDP oracle ignores theta)."""
    model_tag = cfg.user_model.memory.model
    algo = cfg.user_model.algorithm.name

    name_tag = "mdp" if not cfg.env.use_belief else theta
    run_note = "mdp" if not cfg.env.use_belief else f"model{model_tag}_theta{theta}"
    if cfg.env.use_belief:
        out_dir = os.path.join(cfg.dirpath.model, "user_models", cfg.env.name, f"model{model_tag}")
    else:
        out_dir = os.path.join(cfg.dirpath.model, "user_models", cfg.env.name)
    log_dir = os.path.join(cfg.dirpath.log, "user_model", cfg.env.name, run_note)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = None
    if cfg.logging.wandb_enabled:
        run = wandb.init(project=cfg.logging.wandb_project, group=f"user_model_model{model_tag}",
                         name=run_note, config=OmegaConf.to_container(cfg, resolve=True),
                         sync_tensorboard=True, reinit=True)

    env = make_env(cfg, theta)
    policy_type = cfg.user_model.algorithm[algo].policy_type
    if cfg.env.extractor == "default":
        model = ALGOS[algo](policy_type, env, verbose=1, tensorboard_log=log_dir,
                            device=cfg.training.device, seed=cfg.utils.seed)
    else:
        policy_kwargs = dict(features_extractor_class=EXTRACTORS[cfg.env.extractor],
                             features_extractor_kwargs=dict())
        model = ALGOS[algo](policy_type, env, policy_kwargs=policy_kwargs, verbose=1,
                            tensorboard_log=log_dir, device=cfg.training.device, seed=cfg.utils.seed)

    num_best = cfg.training.num_best_models
    top = []   # (reward, checkpoint_path), highest reward first -- the top-N kept on disk
    for t in range(0, cfg.training.timesteps + 1, cfg.training.eval_freq):
        model.learn(total_timesteps=cfg.training.eval_freq, reset_num_timesteps=False)
        mean_reward, _ = evaluate_policy(model, model.get_env(), cfg.training.n_eval_episodes)
        ckpt = os.path.join(out_dir, f"ppo_{name_tag}_{model.num_timesteps}.zip")
        model.save(ckpt)
        top.append((mean_reward, ckpt))
        top.sort(key=lambda x: -x[0])
        while len(top) > num_best: 
            _, worst = top.pop()
            if os.path.exists(worst):
                os.remove(worst)
        print(f"[{run_note}] step {t} reward {mean_reward:.3f} top{num_best} {[round(r, 3) for r, _ in top]}", flush=True)
        if run is not None:
            run.log({"eval/mean_reward": mean_reward, "eval/best_reward": top[0][0]}, step=t)

    best_reward, best_ckpt = top[0]
    shutil.copy(best_ckpt, os.path.join(out_dir, f"ppo_{name_tag}_best.zip"))
    print(f"saved top-{len(top)} + ppo_{name_tag}_best.zip (best {best_reward:.3f}) -> {out_dir}")
    if run is not None:
        run.finish()


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.env.use_belief and cfg.user_model.memory.thetas:
        thetas = [float(t) for t in cfg.user_model.memory.thetas]
    else:
        thetas = [cfg.user_model.memory.decay_rate]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None and len(thetas) > 1:
        task = int(task)
        if task >= len(thetas):
            print(f"array task {task} >= {len(thetas)} thetas -- nothing to do")
            return
        thetas = [thetas[task]]

    for theta in thetas:
        train_one(cfg, theta)


if __name__ == "__main__":
    main()
