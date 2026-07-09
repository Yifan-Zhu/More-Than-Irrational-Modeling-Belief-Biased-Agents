import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from stable_baselines3 import PPO
from crci_mem.envs.assistant_env import AdaptiveAssistanceEnv
from crci_mem.user_model.agent_forgetful import ForgetfulHH
from crci_mem.inference.nested_particle_filter import AssistantParticleFilter
import pandas as pd
import pickle
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

def evaluate_assistant(cfg: DictConfig, save_data_path: str):
    """
    Evaluate the AI assistant and save results to pkl file.
    
    Args:
        cfg: Configuration object
        save_data_path: Path to save the evaluation data
    """
    print("Starting evaluation phase...")
    print(OmegaConf.to_yaml(cfg))
    
    if torch.cuda.is_available() and cfg.utils.device == 'cuda':
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    np.random.seed(cfg.utils.seed)
    model_dir = os.path.join(cfg.dirpath.model, 'ai_assistant')
    best_model = os.path.join(model_dir, f"{cfg.assistant.policy_type}_best.zip")
    if not os.path.exists(best_model):
        print(f"No best assistant model at {best_model}.")
        return None
    print(f"Evaluating model: {best_model}")
    
    policy_dir = os.path.join(cfg.dirpath.policy_model, 'user_models', cfg.env.name, f"model{cfg.agent.policy_model}")
    if not os.path.isdir(policy_dir):
        print(f"No Model {cfg.agent.policy_model} user policies at {policy_dir}.")
        return None
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
        print(f"No per-theta policies in {policy_dir}.")
        return None

    thetas_to_evaluate = sorted(list(theta2policy_dir.keys()))
    print(f"Evaluating on theta values: {thetas_to_evaluate}")

    optimal_policy_path = None
    if hasattr(cfg.assistant, 'optimal_policy') and hasattr(cfg.assistant.optimal_policy, 'mdp_file'):
        mdp_filename = cfg.assistant.optimal_policy.mdp_file
        mdp_policy_dir = os.path.join(cfg.dirpath.policy_model, 'user_models', cfg.env.mdp_name)
        optimal_policy_path = os.path.join(mdp_policy_dir, mdp_filename)
        if not os.path.exists(optimal_policy_path):
            print(f"Warning: Optimal MDP policy not found at {optimal_policy_path}")
            optimal_policy_path = None
    
    simulated_agent_params = {
        'env': None,
        'model_cls': PPO,
        'save_dir': '',
        'seed': cfg.utils.seed,
        'device': device,
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
        'device': device,
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
        device=device,
        seed=cfg.utils.seed,
        optimal_policy_path=optimal_policy_path,
        episodes_per_agent=cfg.assistant.episodes_per_agent,
        entropy_threshold=cfg.assistant.entropy_threshold,
        temperature=cfg.agent.temperature,
        deterministic=cfg.agent.deterministic,
        memory_model=cfg.agent.memory_model,
    )
    
    assistant_model = PPO.load(best_model, env=assist_env, device=device)
    evaluation_data = {}
    for theta in thetas_to_evaluate:
        print(f"\nEvaluating on agent with theta={theta}")
        evaluation_data[theta] = {
            'episodes': [],
            'cumulative_rewards': [],
            'assistance_counts': {0: 0, 1: 0, 2: 0},  
            'theta_posterior_history': [],
            'assistance_sequence': []
        }
        
        for episode in range(20):
            obs, info = assist_env.reset(theta=theta, seed=cfg.utils.seed + episode)
            
            done = False
            episode_data = {
                'step': [],
                'reward': [],
                'assistance_type': [],
                'theta_posterior': [],
                'theta_map': [],
                'theta_mean': []
            }
            
            episode_reward = 0
            step = 0
            
            while not done:
                action, _ = assistant_model.predict(obs, deterministic=True)
                action = int(action)
                obs, reward, terminated, truncated, info = assist_env.step(action)
                episode_data['step'].append(step)
                episode_data['reward'].append(reward)
                episode_data['assistance_type'].append(int(action))

                theta_posterior = obs['theta_posterior']
                episode_data['theta_posterior'].append(theta_posterior.copy())
                
                theta_values = np.array(list(theta2policy_dir.keys()))
                map_theta_idx = np.argmax(theta_posterior)
                map_theta = theta_values[map_theta_idx]
                mean_theta = np.sum(theta_values * theta_posterior)
                
                episode_data['theta_map'].append(map_theta)
                episode_data['theta_mean'].append(mean_theta)
                
                evaluation_data[theta]['assistance_counts'][action] += 1
                evaluation_data[theta]['assistance_sequence'].append(action)
                episode_reward += reward
                
                done = terminated or truncated
                step += 1
            
            evaluation_data[theta]['episodes'].append(episode_data)
            evaluation_data[theta]['cumulative_rewards'].append(episode_reward)
            evaluation_data[theta]['theta_posterior_history'].append({
                'map': episode_data['theta_map'],
                'mean': episode_data['theta_mean']
            })
            
            print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}")
        print(f"  Average reward: {np.mean(evaluation_data[theta]['cumulative_rewards']):.2f}")
        assistance_counts = evaluation_data[theta]['assistance_counts']
        total_assists = sum(assistance_counts.values())
        if total_assists > 0:
            print(f"  Assistance distribution:")
            print(f"    No assist: {assistance_counts[0]/total_assists*100:.1f}%")
            print(f"    Action hint: {assistance_counts[1]/total_assists*100:.1f}%")
            print(f"    Memory hint: {assistance_counts[2]/total_assists*100:.1f}%")
    
    data_to_save = {
        'evaluation_data': evaluation_data,
        'thetas_to_evaluate': thetas_to_evaluate,
        'theta2policy_dir': theta2policy_dir,
        'config': OmegaConf.to_yaml(cfg)
    }
    
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
    with open(save_data_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"\nEvaluation data saved to {save_data_path}")
    return evaluation_data


def plot_results(data_path: str, save_dir: str):
    """
    Load evaluation data and create plots.
    
    Args:
        data_path: Path to the saved evaluation data pkl file
        save_dir: Directory to save the plots
    """
    print("Starting plotting phase...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    evaluation_data = data['evaluation_data']
    thetas_to_evaluate = data['thetas_to_evaluate']
    theta2policy_dir = data['theta2policy_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creating plots in {save_dir}")

    plt.figure(figsize=(12, 6))
    x = np.arange(len(thetas_to_evaluate))
    width = 0.25

    no_assist_pcts = []
    action_hint_pcts = []
    memory_hint_pcts = []
    
    for theta in thetas_to_evaluate:
        counts = evaluation_data[theta]['assistance_counts']
        total = sum(counts.values())
        if total > 0:
            no_assist_pcts.append(counts[0]/total*100)
            action_hint_pcts.append(counts[1]/total*100)
            memory_hint_pcts.append(counts[2]/total*100)
        else:
            no_assist_pcts.append(0)
            action_hint_pcts.append(0)
            memory_hint_pcts.append(0)
    
    plt.bar(x - width, no_assist_pcts, width, label='No Assistance')
    plt.bar(x, action_hint_pcts, width, label='Action Hint')
    plt.bar(x + width, memory_hint_pcts, width, label='Memory Hint')
    
    plt.xlabel('Agent Memory Decay (θ)')
    plt.ylabel('Percentage of Steps (%)')
    plt.title('AI Assistant Strategy by Agent Type')
    plt.xticks(x, [f'{t:.1f}' for t in thetas_to_evaluate])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'assistance_distribution.png'))
    plt.close()
    
    for theta in thetas_to_evaluate:
        plt.figure(figsize=(12, 6))
        all_steps = []
        all_map_values = []
        all_mean_values = []
        episode_markers = []
        
        step_counter = 0
        for ep_idx, ep_data in enumerate(evaluation_data[theta]['episodes']):
            steps = ep_data['step']
            steps_offset = [s + step_counter for s in steps]
            all_steps.extend(steps_offset)
            all_map_values.extend(ep_data['theta_map'])
            all_mean_values.extend(ep_data['theta_mean'])
        
            if ep_idx > 0:
                episode_markers.append(step_counter)
            
            step_counter = all_steps[-1] + 1
        
        plt.plot(all_steps, all_map_values, 'b-', label='MAP Theta')
        plt.plot(all_steps, all_mean_values, 'r-', label='Mean Theta')
        plt.axhline(y=theta, color='g', linestyle='--', label='True Theta')
        
        for marker in episode_markers:
            plt.axvline(x=marker, color='k', linestyle=':', alpha=0.5)
        
        plt.xlabel('Timestep')
        plt.ylabel('Theta Value')
        plt.title(f'Theta Posterior Evolution (Agent θ={theta:.1f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'theta_posterior_evolution_{theta:.1f}.png'))
        plt.close()
    
    plt.figure(figsize=(15, 3*len(thetas_to_evaluate)))
    
    for i, theta in enumerate(thetas_to_evaluate):
        assistance_seq = np.array(evaluation_data[theta]['assistance_sequence'])
        colors = np.array(['lightgray', 'lightblue', 'lightgreen'])
        plt.subplot(len(thetas_to_evaluate), 1, i+1)
        plt.scatter(range(len(assistance_seq)), [0]*len(assistance_seq), 
                  c=colors[assistance_seq], marker='s', s=100)

        plt.yticks([])
        plt.title(f'Assistance Sequence (θ={theta:.1f})')
        
        legend_elements = [
            Patch(facecolor='lightgray', label='No Assistance'),
            Patch(facecolor='lightblue', label='Action Hint'),
            Patch(facecolor='lightgreen', label='Memory Hint')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        step_counts = [len(ep_data['step']) for ep_data in evaluation_data[theta]['episodes']]
        boundary = 0
        for count in step_counts[:-1]:
            boundary += count
            plt.axvline(x=boundary-0.5, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'assistance_sequences.png'))
    plt.close()
    
    print(f"All plots saved to {save_dir}")

def plot_assistance_strategy(data_path: str, save_dir: str):
    """
    Loads data and creates a publication-quality bar plot with a
    semi-transparent fill and solid edge color style.
    """
    print("Creating final assistance strategy plot...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    evaluation_data = data['evaluation_data']
    thetas_to_evaluate = data['thetas_to_evaluate']
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    
    palette = sns.color_palette("deep")
    COLOR_NO_ASSIST = palette[0]  # Bright Blue
    COLOR_ACTION = palette[1]     # Bright Orange
    COLOR_MEMORY = palette[2]     # Bright Green

    LABEL_FONTSIZE = 19
    TICK_FONTSIZE = 16
    TITLE_FONTSIZE = 22
    LEGEND_FONTSIZE = 19

    no_assist_pcts, action_hint_pcts, memory_hint_pcts = [], [], []
    for theta in thetas_to_evaluate:
        counts = evaluation_data[theta]['assistance_counts']
        total = sum(counts.values())
        if total > 0:
            no_assist_pcts.append(counts[0] / total * 100)
            action_hint_pcts.append(counts[1] / total * 100)
            memory_hint_pcts.append(counts[2] / total * 100)
        else:
            no_assist_pcts.append(0)
            action_hint_pcts.append(0)
            memory_hint_pcts.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(thetas_to_evaluate)) 
    width = 0.28 
    

    rects1 = ax.bar(x - width, no_assist_pcts, width, 
                    label='No Assistance',
                    facecolor=to_rgba(COLOR_NO_ASSIST, alpha=0.8), 
                    edgecolor=COLOR_NO_ASSIST, 
                    linewidth=1.5)

    rects2 = ax.bar(x, action_hint_pcts, width, 
                    label='Action Hint',
                    facecolor=to_rgba(COLOR_ACTION, alpha=0.8),
                    edgecolor=COLOR_ACTION,
                    linewidth=1.5)

    rects3 = ax.bar(x + width, memory_hint_pcts, width, 
                    label='Memory Hint',
                    facecolor=to_rgba(COLOR_MEMORY, alpha=0.8),
                    edgecolor=COLOR_MEMORY,
                    linewidth=1.5)

    ax.set_xlabel(r'User Memory Decay Rate ($\theta$)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Percentage of Assistance Type (%)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Learned Adaptive Assistance Strategy', fontsize=TITLE_FONTSIZE, pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.1f}' for t in thetas_to_evaluate])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_ylim(0, 105)
    
    ax.legend(fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'assistance_distribution_final.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Final assistance strategy plot saved to {save_dir}")

def plot_assistance_sequences_final(data_path: str, save_dir: str):
    """
    Loads data and creates a publication-quality plot of assistance sequences,
    with accurately plotted episode boundaries.
    """
    print("Creating final assistance sequence plot with accurate boundaries...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    evaluation_data = data['evaluation_data']

    thetas_to_plot = [t for t in (0.0, 0.4, 0.6, 1.0) if t in evaluation_data] or sorted(evaluation_data)
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    
    palette = sns.color_palette("deep")
    COLOR_ACTION = palette[0]  # Bright Blue
    COLOR_MEMORY = palette[1]  # Bright Orange
    COLOR_NO_ASSIST = 'lightgray'
    cmap = {0: COLOR_NO_ASSIST, 1: COLOR_ACTION, 2: COLOR_MEMORY}

    LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 16
    TITLE_FONTSIZE = 23
    ANNOTATION_FONTSIZE = 16
    LEGEND_FONTSIZE = 19

    n_subplots = len(thetas_to_plot)
    fig, axs = plt.subplots(n_subplots, 1,
                            figsize=(10, 1.5 * n_subplots + 1),
                            sharex=False)
    axs = np.atleast_1d(axs)   

    for i, theta in enumerate(thetas_to_plot):
        ax = axs[i]
        all_assistance_seq = np.concatenate([ep['assistance_type'] for ep in evaluation_data[theta]['episodes']])
        timesteps = np.arange(len(all_assistance_seq))
        
        colors = [cmap[val] for val in all_assistance_seq]
        ax.scatter(timesteps, np.zeros_like(timesteps), 
                   c=colors, marker='s', s=100)
        
        boundary_position = 0
        episode_lengths = [len(ep['assistance_type']) for ep in evaluation_data[theta]['episodes']]
        
        for length in episode_lengths[:-1]:
            boundary_position += length
            ax.axvline(x=boundary_position - 0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 5)) 

        if i == len(thetas_to_plot) - 1:
            ax.set_xlabel('Timesteps', fontsize=LABEL_FONTSIZE)
        
        ax.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)
        
        ax.text(0.02, 0.5, fr'$\theta={theta:.1f}$', 
                transform=ax.transAxes, 
                fontsize=ANNOTATION_FONTSIZE, 
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.7))

    legend_elements = [
        Patch(facecolor=COLOR_NO_ASSIST, label='No Assistance'),
        Patch(facecolor=COLOR_ACTION, label='Action Hint'),
        Patch(facecolor=COLOR_MEMORY, label='Memory Hint')
    ]
    fig.legend(handles=legend_elements, 
               loc='upper right', bbox_to_anchor=(0.98, 0.92), 
               fontsize=LEGEND_FONTSIZE, ncol=3)

    fig.suptitle('Assistance Sequences for Selective Users', fontsize=TITLE_FONTSIZE, y=0.96)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(save_dir, 'assistance_sequences.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Final assistance sequence plot with corrected boundaries saved to {save_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="assistant_config.yaml")
def main(cfg: DictConfig):
    """
    Main function that can run evaluation, plotting, or both.
    Use evaluation.mode parameter in config to control behavior:
    - evaluation.mode: evaluate (default) - Run evaluation and save data
    - evaluation.mode: plot - Load data and create plots only
    - evaluation.mode: both - Run evaluation then create plots
    - evaluation.data_path: Custom path to data file (optional)
    """
    mode = cfg.evaluation.mode
    custom_data_path = cfg.evaluation.data_path

    base_save_dir = os.path.join(cfg.dirpath.result, 'ai_assistant')
    data_path = custom_data_path if custom_data_path else os.path.join(base_save_dir, 'evaluation_data.pkl')
    plot_dir = os.path.join(base_save_dir, 'figures')
    
    print(f"Running in mode: {mode}")
    
    if mode in ['evaluate', 'both']:
        evaluation_data = evaluate_assistant(cfg, data_path)
        if evaluation_data is None:
            print("Evaluation failed, exiting.")
            return
    
    if mode in ['plot', 'both']:
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Please run evaluation first.")
            return
        # Paper Fig-5: the learned adaptive assistance strategy.
        plot_assistance_strategy(data_path, plot_dir)
        # Paper Fig-6: assistance timing sequences for selective users.
        plot_assistance_sequences_final(data_path, plot_dir)
    
    print(f"\nCompleted! Results in {base_save_dir}")


if __name__ == "__main__":
    main() 