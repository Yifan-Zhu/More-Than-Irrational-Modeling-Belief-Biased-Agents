"""
Curriculum warm-start for the user policies.

Seed from the theta=0.0 policy and fine-tune at increasing theta, each level warm-starting from the
previous one, so behaviour grows gradually rather than collapsing into a theta-agnostic rush-and-guess
local optimum, which is what COLD training does at mid/high theta.
"""
import argparse
import os
import shutil

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import crci_mem 
from crci_mem.user_model.agent_forgetful import make_belief_wrapper


def make_env(theta, mem_model="B", env_name="MemoryDecayExp_9x9-v0"):
    return make_belief_wrapper(gym.make(env_name, render_mode="rgb_array"), theta, mem_model)


def finetune(prev_policy, theta, steps, eval_freq, n_eval, device, run_dir, wandb_run, mem_model="B", num_best=5):
    """Warm-start from prev_policy, fine-tune at theta; return the top-N checkpoints (best first)."""
    os.makedirs(run_dir, exist_ok=True)
    model = PPO.load(prev_policy, env=make_env(theta, mem_model), device=device)
    model.tensorboard_log = run_dir
    top = []   # (reward, path), highest reward first
    for t in range(0, steps + 1, eval_freq):
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=(t == 0), tb_log_name=f"theta{theta}")
        mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval)
        ckpt = os.path.join(run_dir, f"ckpt_{model.num_timesteps}.zip")
        model.save(ckpt)
        top.append((mean_reward, ckpt))
        top.sort(key=lambda x: -x[0])
        while len(top) > num_best:
            _, worst = top.pop()
            if os.path.exists(worst):
                os.remove(worst)
        print(f"[curriculum {mem_model} theta={theta}] step {t} reward {mean_reward:.3f} "
              f"top{num_best} {[round(r, 3) for r, _ in top]}", flush=True)
        if wandb_run is not None:
            wandb_run.log({"eval/mean_reward": mean_reward}, step=t)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-policy", required=True, help="theta=0.0 policy that seeds the curriculum")
    ap.add_argument("--model", default="B", choices=["A", "B"], help="memory model to warm-start")
    ap.add_argument("--thetas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--steps-per-theta", type=int, default=400000)
    ap.add_argument("--eval-freq", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--num-best", type=int, default=5, help="top-N eval checkpoints to keep per theta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None,
                    help="policy output dir; defaults to runs/default/models/user_models/<env>/model<A|B>")
    ap.add_argument("--log-dir", default="runs/default/logs/user_model/curriculum",
                    help="scratch for per-theta fine-tuning checkpoints + tensorboard (kept out of models/)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or f"runs/default/models/user_models/MemoryDecayExp_9x9-v0/model{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    thetas = [float(t) for t in args.thetas.split(",")]

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="crci-memory", group=f"curriculum_model{args.model}", name="curriculum",
                         config=vars(args), sync_tensorboard=True, reinit=True)

    # theta=0.0 == no decay, so the optimal is the theta=0.0 policy
    seed_dst = os.path.join(out_dir, "ppo_0.0_best.zip")
    if os.path.abspath(args.init_policy) != os.path.abspath(seed_dst):
        shutil.copy(args.init_policy, seed_dst)
    prev = args.init_policy
    for theta in thetas:
        top = finetune(prev, theta, args.steps_per_theta, args.eval_freq, args.n_eval, args.device,
                       os.path.join(args.log_dir, f"run_theta_{theta}"), run, args.model, args.num_best)
        if not top:
            print(f"[theta={theta}] no checkpoint produced -- stopping")
            break
        # keep the top-N candidates (evaluate_and_select re-picks) + the canonical best; chain from best
        for _, ckpt in top:
            steps_tag = os.path.basename(ckpt)[len("ckpt_"):-len(".zip")]
            shutil.copy(ckpt, os.path.join(out_dir, f"ppo_{theta}_{steps_tag}.zip"))
        shutil.copy(top[0][1], os.path.join(out_dir, f"ppo_{theta}_best.zip"))
        print(f"[theta={theta}] -> {out_dir}/ppo_{theta}_best.zip (+{len(top)} candidates)")
        prev = os.path.join(out_dir, f"ppo_{theta}_best.zip")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
