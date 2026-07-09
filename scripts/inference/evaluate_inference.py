"""
Evaluate a finished inference run and draw the paper figures, reusing the pipeline plotting fns:
  - plot_error_paper_2(df, temp): aggregated (over seeds×θ) PM/MAP error vs step with SE bands,
    annotating the 90% / 95% (/99%) error-reduction points.
  - posterior evolution per θ (seed 66), BOTH static (5-timestep P(θ|actions) panel) and animated GIF.
  - with --trajectories: the actual-user (τ=3) trajectory per θ (seed 66), BOTH static and animated.
"""
import argparse
import os

import numpy as np
import pandas as pd

import crci_mem  
from crci_mem.pipeline.pipeline import InferencePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/default/results/inference/modelA",
                    help="inference run dir (contains raw/); use .../modelB for a Model B user")
    ap.add_argument("--temp", type=float, default=3.0, help="temperature slice to aggregate (paper = 3.0)")
    ap.add_argument("--trajectories", action="store_true",
                    help="also render per-theta actual-user (tau=3) trajectories, static + animated (seed 66)")
    args = ap.parse_args()

    results_dir = os.path.join(args.run_dir, "raw")
    data_dir = os.path.join(args.run_dir, "data")
    plot_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    pipe = InferencePipeline(save_dir=plot_dir)
    pipe.data_processing_streaming(results_dir, data_dir)
    data_df = pd.read_pickle(os.path.join(data_dir, "data.pkl"))
    n_exp = data_df[["theta_true", "seed"]].drop_duplicates().shape[0]
    print(f"loaded {len(data_df)} per-step rows from {n_exp} experiments "
          f"(theta={sorted(np.unique(data_df['theta_true']))}, temps={sorted(np.unique(data_df['temp']))})")

    df = data_df[data_df["temp"] == args.temp].reset_index(drop=True)
    pipe.plot_error_paper_2(df, temp=args.temp)
    print(f"aggregated-error figure (90/95 reduction) -> {plot_dir}/")

    for theta in np.unique(data_df["theta_true"].values):
        sub = df[(df["theta_true"] == theta) & (df["seed"] == 66)].reset_index(drop=True)
        if sub.empty:
            continue
        pipe.plot_static_posterior(theta_particles_history=sub, theta_true=theta, temp=args.temp)
        pipe.plot_animated_theta_posterior(theta_particles_history=sub, theta_true=theta, temp=args.temp)
    print(f"per-theta posterior evolution (static + animated) -> {plot_dir}/")

    if args.trajectories:
        env_traj_df = pd.read_pickle(os.path.join(data_dir, "env_traj.pkl"))
        for theta in np.unique(data_df["theta_true"].values):
            sub = env_traj_df[(env_traj_df["theta_true"] == theta) & (env_traj_df["seed"] == 66)]
            if sub.empty:
                continue
            item = sub.iloc[0]["env_trajectory"][0]   # {'env', 'trajectory'}
            pipe.plot_trajectory(theta=theta, env=item["env"].unwrapped,
                                 traj=item["trajectory"], filename=f"inf_theta_{theta}_seed66")
            pipe.plot_animated_trajectory(theta=theta, env=item["env"].unwrapped,
                                          traj=item["trajectory"], filename=f"inf_theta_{theta}_seed66")
        print(f"per-theta trajectories (static + animated) -> {plot_dir}/trajectory/")


if __name__ == "__main__":
    main()
