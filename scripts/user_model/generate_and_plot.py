"""
Roll out a trained per-theta policy under Model A or Model B, save the trajectories, and render
the env trajectory figures.
Renders the per-theta trajectory figures under Model A and Model B (uses ppo_<theta>_best.zip):

        python scripts/user_model/generate_and_plot.py --model A --gif
        python scripts/user_model/generate_and_plot.py --model B --gif
        # defaults: --policy-dir runs/default/models/user_models/<env>, --out-dir runs/default/results/user_model
"""
import argparse
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO

import crci_mem  
from crci_mem.user_model.wrappers import BeliefWrapper, PersistentBeliefWrapper
from crci_mem.pipeline.pipeline import InferencePipeline


def make_wrapper(model, env, theta):
    if model == "A":
        return BeliefWrapper(env, decay=theta, memory_decay_model="rand_drop")
    return PersistentBeliefWrapper(env, decay=theta)


def rollout(policy, wrapped, env, max_steps, deterministic):
    obs, _ = wrapped.reset(env=env)
    u = wrapped.env  
    traj, done, steps, reward = [], False, 0, 0.0
    while not done and steps < max_steps:
        s = u._get_state()          
        o = u._get_obs()            
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = wrapped.step(action)
        traj.append({"s": s, "o": o, "m": None, "a": action, "b": obs, "reward": reward})
        done = terminated or truncated
        steps += 1
    traj.append({"s": u._get_state(), "o": u._get_obs(), "m": None, "a": None, "b": None, "reward": reward})
    return traj, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B"], required=True)
    ap.add_argument("--policy-dir", default="runs/default/models/user_models/MemoryDecayExp_9x9-v0",
                    help="env policy dir; the model<A|B>/ subdir is chosen by --model")
    ap.add_argument("--policy-pattern", default="ppo_{theta}_best.zip",
                    help="e.g. 'ppo_{theta}_best.zip' or 'ppo_{theta}_<steps>.zip'")
    ap.add_argument("--thetas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--episodes", type=int, default=1, help="trajectories per theta (episode 0 is plotted)")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax actions; default is native sampling")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out-dir", default="runs/default/results/user_model")
    ap.add_argument("--gif", action="store_true", help="also render the animated GIF per theta")
    args = ap.parse_args()

    print(f"crci_mem from: {os.path.dirname(crci_mem.__file__)}")
    os.makedirs(args.out_dir, exist_ok=True)
    thetas = [float(t) for t in args.thetas.split(",")]
    pipe = InferencePipeline(save_dir=args.out_dir)  
    env = gym.make("MemoryDecayExp_9x9-v0", render_mode="rgb_array")

    pol_dir = os.path.join(args.policy_dir, f"model{args.model}")   
    records = []
    for theta in thetas:
        ppath = os.path.join(pol_dir, args.policy_pattern.format(theta=theta))
        if not os.path.exists(ppath):
            print(f"[skip] missing policy: {ppath}")
            continue
        policy = PPO.load(ppath, device=args.device)
        wrapped = make_wrapper(args.model, env, theta)
        first_traj = None
        for ep in range(args.episodes):
            env.reset(seed=args.seed0 + ep)  
            traj, reward = rollout(policy, wrapped, env, args.max_steps, args.deterministic)
            records.append({"theta_true": theta, "seed": args.seed0 + ep, "trajectory": traj})
            if ep == 0:
                first_traj = traj
        if first_traj is not None:
            pipe.plot_trajectory(theta=theta, env=env.unwrapped, traj=first_traj,
                                 filename=f"model{args.model}_theta{theta}")
            if args.gif:
                pipe.plot_animated_trajectory(theta=theta, env=env.unwrapped, traj=first_traj,
                                              filename=f"model{args.model}_theta{theta}")
            print(f"[model {args.model} theta={theta}] len={len(first_traj)} plotted -> {args.out_dir}/trajectory/")

    out_pkl = os.path.join(args.out_dir, f"trajectories_model{args.model}.pkl")
    pd.DataFrame(records).to_pickle(out_pkl)
    print(f"saved {len(records)} trajectories -> {out_pkl}")


if __name__ == "__main__":
    main()
