"""
Validate the retrained Model A user policies, pick the best available per theta from everything on
disk (no re-training), and render a representative trajectory per theta.
Each candidate is evaluated over N episodes; the best by native success (tie: mean reward) is copied to the
canonical ppo_<theta>_best.zip and its representative trajectory is rendered with the paper's
plot_trajectory. 
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import crci_mem  
from crci_mem.pipeline.pipeline import InferencePipeline
from generate_and_plot import make_wrapper, rollout


def gather_candidates(policy_dir, theta):
    """Every on-disk policy that could be this theta's user model, most-preferred first."""
    pats = [
        f"ppo_{theta}_seed*_best.zip",
        f"ppo_{theta}_seed*_[0-9]*.zip",
        f"ppo_{theta}_best.zip",
        f"ppo_{theta}_[0-9]*.zip",
    ]
    out = []
    for p in pats:
        out += sorted(glob.glob(os.path.join(policy_dir, p)))
    seen, uniq = set(), []
    for c in out:
        a = os.path.abspath(c)
        if a not in seen:
            seen.add(a)
            uniq.append(c)
    return uniq


def label(path):
    orig = "ORIG:" if "user_models" in path else ""
    b = os.path.basename(path)
    m = re.search(r"_seed([^_]+)_best\.zip$", b)
    if m:
        return orig + f"s{m.group(1)}best"
    m = re.search(r"_seed([^_]+)_\d+\.zip$", b)
    if m:
        return orig + f"s{m.group(1)}fin"
    m = re.search(r"ppo_(\d+)\.zip$", b)
    if m:
        sd = re.search(r"run_theta_[0-9.]+_seed([^/]+)/", path)
        return orig + (f"s{sd.group(1)}@" if sd else "ck@") + f"{int(m.group(1))//1000}k"
    return orig + b.replace(".zip", "")


def eval_policy(policy_path, model, theta, env, episodes, max_steps, seed0, device):
    policy = PPO.load(policy_path, device=device)
    wrapped = make_wrapper(model, env, theta)
    succ, rew, length, trajs = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        traj, reward = rollout(policy, wrapped, env, max_steps, deterministic=False)  
        succ.append(bool(reward > 0)); rew.append(reward); length.append(len(traj) - 1); trajs.append(traj)
    return {"success": float(np.mean(succ)), "reward": float(np.mean(rew)),
            "length": float(np.mean(length)), "successes": succ, "trajs": trajs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A", choices=["A", "B"])
    ap.add_argument("--policy-dir", default="runs/default/models/user_models/MemoryDecayExp_9x9-v0",
                    help="env policy dir; the model<A|B>/ subdir is chosen by --model")
    ap.add_argument("--extra-policy-dir", default=None,
                    help="also consider policies in this dir as candidates")
    ap.add_argument("--thetas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--episodes", type=int, default=30, help="native episodes per candidate")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out-dir", default="runs/default/results/user_model")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    print(f"crci_mem from: {os.path.dirname(crci_mem.__file__)}")
    os.makedirs(args.out_dir, exist_ok=True)
    thetas = [float(t) for t in args.thetas.split(",")]
    pipe = InferencePipeline(save_dir=args.out_dir)
    env = gym.make("MemoryDecayExp_9x9-v0", render_mode="rgb_array")

    pol_dir = os.path.join(args.policy_dir, f"model{args.model}")   # the model<A|B>/ subdir
    rows = []
    print(f"\n{'theta':>6} {'winner':>10} {'succ':>6} {'reward':>7} {'len':>6} {'#cand':>6}   candidates (succ)")
    print("-" * 92)
    dirs = [pol_dir] + ([args.extra_policy_dir] if args.extra_policy_dir else [])
    for theta in thetas:
        cands = []
        for d in dirs:
            cands += gather_candidates(d, theta)
        seen = set()
        cands = [c for c in cands if not (os.path.abspath(c) in seen or seen.add(os.path.abspath(c)))]
        if not cands:
            print(f"{theta:>6}   (no policy found)")
            continue

        scored = []
        for z in cands:
            try:
                r = eval_policy(z, args.model, theta, env, args.episodes, args.max_steps, args.seed0, args.device)
            except Exception as e:
                print(f"       [skip {label(z)}: {type(e).__name__}]")
                continue
            scored.append((r, z))
        if not scored:
            print(f"{theta:>6}   (all candidates failed to load/eval)")
            continue

        scored.sort(key=lambda x: (x[0]["success"], x[0]["reward"]), reverse=True)
        best, best_zip = scored[0]
        dst = os.path.join(pol_dir, f"ppo_{theta}_best.zip")
        if os.path.abspath(best_zip) != os.path.abspath(dst):
            shutil.copy(best_zip, dst)

        # Plot a FIXED sample episode (episode 0). 
        rep_idx = 0
        env.reset(seed=args.seed0 + rep_idx)
        pipe.plot_trajectory(theta=theta, env=env.unwrapped, traj=best["trajs"][rep_idx],
                             filename=f"model{args.model}_theta{theta}")
        if args.gif:
            pipe.plot_animated_trajectory(theta=theta, env=env.unwrapped, traj=best["trajs"][rep_idx],
                                          filename=f"model{args.model}_theta{theta}")

        allc = " ".join(f"{label(z)}={r['success']:.2f}" for r, z in scored[:8])
        print(f"{theta:>6} {label(best_zip):>10} {best['success']:>6.2f} {best['reward']:>7.3f} "
              f"{best['length']:>6.1f} {len(scored):>6}   {allc}")
        rows.append({"theta": theta, "winner": os.path.basename(best_zip), "native_success": best["success"],
                     "mean_reward": best["reward"], "mean_length": best["length"], "n_candidates": len(scored)})

    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nselected best -> {pol_dir}/ppo_<theta>_best.zip")
    print(f"figures       -> {args.out_dir}/trajectory/model{args.model}_theta*.png")
    print(f"summary       -> {args.out_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
