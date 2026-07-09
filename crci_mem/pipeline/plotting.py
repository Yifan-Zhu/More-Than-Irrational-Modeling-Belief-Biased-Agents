import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_trajectory(df=None, theta=None, env=None, traj=None, filename=None, save_dir=''):
    def composite_colormap(base_cmap_name, alpha, bg_color):
        base_cmap = plt.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        bg_color = np.array(bg_color)
        composite_colors = alpha * colors + (1 - alpha) * bg_color
        return ListedColormap(composite_colors)
    if env is not None and traj is not None:
        env = env
        traj = traj
    else:
        env = df['env'].values[0].unwrapped
        traj = df['trajectory'].values[0]
    bg_img = env.render_grid()
    tile_size = env.tile_size
    grid_size = env.grid_size
    avg_bg_color = bg_img.mean(axis=(0,1)) / 255.0  
    avg_bg_color = np.append(avg_bg_color, 1)         
    visit_freq = np.zeros((grid_size, grid_size))
    for step in traj:
        visit_freq[step['s']['agent_pos'][1], step['s']['agent_pos'][0]] += 1
        #visit_freq[step['s'][1], step['s'][0]] += 1
    TITLE_FONTSIZE = 25
    LABEL_FONTSIZE = 22
    LEGEND_FONTSIZE = 15
    TICK_FONTSIZE = 18
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    ax.imshow(bg_img, origin='upper')
    extent = [0, grid_size * tile_size, grid_size * tile_size, 0]
    heatmap = ax.imshow(visit_freq, cmap='Blues', alpha=0.5, interpolation='nearest',
                        extent=extent, vmin=0, vmax=visit_freq.max())
    divider = make_axes_locatable(ax)
    cax_heat = divider.append_axes("right", size="5%", pad=0.05)
    comp_cmap = composite_colormap('Blues', 0.5, avg_bg_color)
    mappable = plt.cm.ScalarMappable(cmap=comp_cmap, norm=plt.Normalize(vmin=0, vmax=visit_freq.max()))
    mappable.set_array([])
    # the color bar should only show integer values
    ticks = np.arange(0, visit_freq.max() + 1, 1)
    cbar = plt.colorbar(mappable, ax=ax, label='Visit Frequency', cax=cax_heat, ticks=ticks)
    cbar.set_label('Visit Frequency', fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    traj_pixels = np.array([
        [step['s']['agent_pos'][0] * tile_size + tile_size / 2, step['s']['agent_pos'][1] * tile_size + tile_size / 2]
        for step in traj
        #[step['s'][0] * tile_size + tile_size / 2, step['s'][1] * tile_size + tile_size / 2]
        #for step in traj
    ])
    segments = np.concatenate([traj_pixels[:-1, None, :], traj_pixels[1:, None, :]], axis=1)
    lc = LineCollection(segments, cmap='YlOrRd', norm=plt.Normalize(0, len(segments)))
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.plot(traj_pixels[0, 0], traj_pixels[0, 1], marker='*', color='green', markersize=25, markeredgecolor='White', markeredgewidth=2,label='Start')
    ax.plot(traj_pixels[-1, 0], traj_pixels[-1, 1], marker='*', color='red', markersize=25, markeredgecolor='White', markeredgewidth=2, label='End')

    ax.set_xlim(0, grid_size * tile_size)
    ax.set_ylim(grid_size * tile_size, 0)
    ax.set_title(f'Trajectory ($\\theta$ = {theta})', fontsize=TITLE_FONTSIZE, pad=10)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    cax_seg = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar_seg = fig.colorbar(lc, cax=cax_seg, orientation='horizontal')
    cbar_seg.set_label('Timestep', fontsize=LABEL_FONTSIZE)
    cbar_seg.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar_seg.ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.tight_layout(pad=0.5)
    save_path = os.path.join(save_dir, 'trajectory')
    os.makedirs(save_path, exist_ok=True)
    if filename is None:
        file_path = os.path.join(save_path, f'trajectory_theta_{theta}.pdf')
    else:
        file_path = os.path.join(save_path, f'trajectory_{filename}.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_animated_trajectory(theta=None, env=None, traj=None, filename=None, save_dir=''):
    """
    Generate an animated GIF showing the trajectory being built step by step.
    """
    # Check if traj is valid
    if traj is None or len(traj) == 0:
        print(f"Warning: Empty trajectory for theta={theta}, skipping animation")
        return
    
    def composite_colormap(base_cmap_name, alpha, bg_color):
        base_cmap = plt.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        bg_color = np.array(bg_color)
        composite_colors = alpha * colors + (1 - alpha) * bg_color
        return ListedColormap(composite_colors)
    
    bg_img = env.render_grid()
    tile_size = env.tile_size
    grid_size = env.grid_size
    avg_bg_color = bg_img.mean(axis=(0,1)) / 255.0  
    avg_bg_color = np.append(avg_bg_color, 1)
    
    TITLE_FONTSIZE = 25
    LABEL_FONTSIZE = 22
    LEGEND_FONTSIZE = 15
    TICK_FONTSIZE = 18
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    
    def get_pos(step):
        """Extract position from step, handling different data structures."""
        if isinstance(step['s'], dict) and 'agent_pos' in step['s']:
            return step['s']['agent_pos']
        elif isinstance(step['s'], (list, np.ndarray)) and len(step['s']) >= 2:
            return step['s'][:2]  # First two elements as [x, y]
        else:
            return step['s']
    
    def animate(i):
        ax.clear()
        ax.axis('off')
        ax.imshow(bg_img, origin='upper')
        extent = [0, grid_size * tile_size, grid_size * tile_size, 0]
        
        visit_freq = np.zeros((grid_size, grid_size))
        traj_up_to_i = traj[:i+1]
        for step in traj_up_to_i:
            pos = get_pos(step)
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                visit_freq[int(pos[1]), int(pos[0])] += 1
        
        if visit_freq.max() > 0:
            heatmap = ax.imshow(visit_freq, cmap='Blues', alpha=0.5, interpolation='nearest',
                                extent=extent, vmin=0, vmax=visit_freq.max())

        if len(traj_up_to_i) > 1:
            traj_pixels = []
            for step in traj_up_to_i:
                pos = get_pos(step)
                if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                    traj_pixels.append([
                        pos[0] * tile_size + tile_size / 2, 
                        pos[1] * tile_size + tile_size / 2
                    ])
            
            if len(traj_pixels) > 1:
                traj_pixels = np.array(traj_pixels)
                segments = np.concatenate([traj_pixels[:-1, None, :], traj_pixels[1:, None, :]], axis=1)
                lc = LineCollection(segments, cmap='YlOrRd', norm=plt.Normalize(0, len(segments)))
                lc.set_array(np.arange(len(segments)))
                lc.set_linewidth(2)
                ax.add_collection(lc)
        
        if len(traj_up_to_i) > 0:
            start_pos = get_pos(traj_up_to_i[0])
            if isinstance(start_pos, (list, np.ndarray)) and len(start_pos) >= 2:
                start_pixel = [
                    start_pos[0] * tile_size + tile_size / 2, 
                    start_pos[1] * tile_size + tile_size / 2
                ]
                ax.plot(start_pixel[0], start_pixel[1], marker='*', color='green', 
                       markersize=25, markeredgecolor='White', markeredgewidth=2, label='Start')
                
                current_pos = get_pos(traj_up_to_i[-1])
                if isinstance(current_pos, (list, np.ndarray)) and len(current_pos) >= 2:
                    current_pixel = [
                        current_pos[0] * tile_size + tile_size / 2,
                        current_pos[1] * tile_size + tile_size / 2
                    ]
                    ax.plot(current_pixel[0], current_pixel[1], marker='o', color='blue',
                           markersize=15, markeredgecolor='White', markeredgewidth=2, label='Current')
        
        ax.set_xlim(0, grid_size * tile_size)
        ax.set_ylim(grid_size * tile_size, 0)
        ax.set_title(f'Trajectory ($\\theta$ = {theta}) - Step {i+1}/{len(traj)}', 
                    fontsize=TITLE_FONTSIZE, pad=10)
        if len(traj_up_to_i) > 0:
            ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
    
    if len(traj) == 0:
        print(f"Warning: Empty trajectory for theta={theta}")
        plt.close(fig)
        return
    
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(traj),
        interval=200,  
        blit=False,
        repeat=True
    )
    
    animate(0)
    plt.draw()
    
    save_path = os.path.join(save_dir, 'trajectory')
    os.makedirs(save_path, exist_ok=True)
    if filename is None:
        animation_file = os.path.join(save_path, f'animated_trajectory_theta_{theta}.gif')
    else:
        animation_file = os.path.join(save_path, f'animated_trajectory_{filename}.gif')
    
    try:
        ani.save(animation_file, writer='pillow', fps=5)
        print(f"Saved animated trajectory to {animation_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print(f"Trajectory length: {len(traj)}")
        if len(traj) > 0:
            print(f"First step structure: {traj[0]}")
    finally:
        plt.close(fig)


def plot_error_paper_2(df_raw, temp=None, save_dir=''):
    # df_raw is the raw dataframe with thousands of rows.
    df_agg = df_raw.groupby('step').agg(
        mean_pm_error=('mean_error', 'mean'),
        se_pm_error=('mean_error', 'sem'), 
        mean_map_error=('map_error', 'mean'),
        se_map_error=('map_error', 'sem')
    ).reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')

    palette = sns.color_palette("deep")
    COLOR_BLUE = palette[0]
    COLOR_ORANGE = palette[1]

    LABEL_FONTSIZE = 16
    TICK_FONTSIZE = 13
    TITLE_FONTSIZE = 18
    ANNOTATION_FONTSIZE = 16
    LEGEND_FONTSIZE = 16

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the mean lines
    ax.plot(df_agg['step'], df_agg['mean_pm_error'], color=COLOR_BLUE, lw=2.5, label='PM Error')
    ax.plot(df_agg['step'], df_agg['mean_map_error'], color=COLOR_ORANGE, lw=2.5, label='MAP Estimate Error')

    # Plot the standard error bands (the shaded area)
    ax.fill_between(df_agg['step'], 
                    df_agg['mean_pm_error'] - df_agg['se_pm_error'], 
                    df_agg['mean_pm_error'] + df_agg['se_pm_error'], 
                    color=COLOR_BLUE, alpha=0.2)
    ax.fill_between(df_agg['step'], 
                    df_agg['mean_map_error'] - df_agg['se_map_error'], 
                    df_agg['mean_map_error'] + df_agg['se_map_error'], 
                    color=COLOR_ORANGE, alpha=0.2)
    # print the last step of the aggregated data
    print(f'last step: {df_agg["step"].iloc[-1]}')
    print(f'mean_pm_error: {df_agg["mean_pm_error"].iloc[-1]}')
    print(f'se_pm_error: {df_agg["se_pm_error"].iloc[-1]}')
    print(f'mean_map_error: {df_agg["mean_map_error"].iloc[-1]}')
    print(f'se_map_error: {df_agg["se_map_error"].iloc[-1]}')

    initial_error = df_agg['mean_pm_error'].iloc[0]
    thresholds = {'90%': initial_error * 0.10,
                '95%': initial_error * 0.05,
                '99%': initial_error * 0.01}
    
    annotation_points = []
    for label, threshold in thresholds.items():
        try:
            # Find the first index ON THE AGGREGATED DATA where the mean error is below the threshold
            converged_df = df_agg[df_agg['mean_pm_error'] <= threshold]
            if not converged_df.empty:
                first_converged_idx = converged_df.index[0]
                step = df_agg['step'].iloc[first_converged_idx]
                error = df_agg['mean_pm_error'].iloc[first_converged_idx]
                annotation_points.append({'step': step, 'error': error, 'label': label})
        except IndexError:
            continue

    y_text_positions = [initial_error * 0.55, initial_error * 0.40, initial_error * 0.25]
    
    if annotation_points:
        annotation_points.sort(key=lambda p: p['step'])
        
        for i, point in enumerate(annotation_points):
            step, error, label = point['step'], point['error'], point['label']

            ax.plot(step, error, 
                    marker='o', markersize=8,
                    markerfacecolor='none',
                    markeredgecolor=COLOR_BLUE,
                    markeredgewidth=1.5)

            text_x = df_agg['step'].max() * 0.55
            text_y = y_text_positions[i]
            ax.annotate("", xy=(step, error), xytext=(text_x, text_y),
                        arrowprops=dict(arrowstyle="-", color="gray", connectionstyle="arc3,rad=0.1"))
            ax.text(text_x + 2, text_y, f'{label} reduction at t={step}', 
                    fontsize=ANNOTATION_FONTSIZE, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.6))

    ax.set_xlabel('Timesteps', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Error', fontsize=LABEL_FONTSIZE)
    ax.set_title('Inference Error Convergence', fontsize=TITLE_FONTSIZE, pad=15)
    
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    ax.set_xlim(0, df_raw['step'].max()+1)
    ax.set_ylim(0) 

    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

    plt.tight_layout()
    
    save_dir = os.path.join(save_dir, 'evolution_error')
    os.makedirs(save_dir, exist_ok=True)
    
    if temp is not None:
        file_name = f'error_convergence_temp_{temp}_v2.pdf'
    else:
        file_name = 'error_convergence_v2.pdf'
    
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)


def plot_static_posterior(theta_particles_history, theta_true, temp=None, save_dir=''):
    """Step-by-step inference posterior over theta. 
    theta_particles_history is a per-step DataFrame with a 'posterior' column ({theta: prob})."""
    sns.set_theme()
    n_entries = len(theta_particles_history)
    if n_entries < 5:
        indices = np.arange(n_entries)
        fig, axs = plt.subplots(n_entries, 1, figsize=(8, 4 * n_entries))
    else:
        indices = np.linspace(0, n_entries - 1, 5, dtype=int)
        fig, axs = plt.subplots(5, 1, figsize=(8, 20))

    for ax, idx in zip(axs, indices):
        entry = theta_particles_history.iloc[idx]
        step_val = entry.get('step', idx)
        posterior_data = entry['posterior']
        df = pd.DataFrame({'theta': list(posterior_data.keys()),
                           'posterior': list(posterior_data.values())})
        df.sort_values('theta', inplace=True)
        try:
            sns.histplot(data=df, x='theta', weights='posterior', bins=11, kde=True,
                         color='skyblue', alpha=0.4, ax=ax)
        except Exception:
            sns.histplot(data=df, x='theta', weights='posterior', bins=11, kde=False,
                         color='skyblue', alpha=0.4, ax=ax)
        mean_weighted_theta = np.average(df['theta'], weights=df['posterior'])
        predicted_theta = df.loc[df['posterior'].idxmax(), 'theta']
        y_min, y_max = ax.get_ylim()
        ax.vlines(theta_true, y_min, y_max, color='blue', linestyle='-', label='True Theta')
        ax.vlines(mean_weighted_theta, y_min, y_max, color='green', linestyle='--', label='Posterior Mean')
        ax.vlines(predicted_theta, y_min, y_max, color='red', linestyle='-', label='MAP Estimate')
        ax.set_title(f"Timestep {step_val + 1}")
        ax.legend()

    if temp is not None:
        fig.suptitle(f'Estimated $P(\\theta | \\tau)$ over Time\n$\\theta$ = {theta_true}, temp={temp}', fontsize=16)
    else:
        fig.suptitle(f'Estimated $P(\\theta | \\tau)$ over Time\n$\\theta$ = {theta_true}', fontsize=16)
    plt.tight_layout()

    save_dir = os.path.join(save_dir, 'static_animation')
    os.makedirs(save_dir, exist_ok=True)
    if temp is not None:
        out_file = os.path.join(save_dir, f'static_animation_theta_{theta_true}_temp_{temp}.png')
    else:
        out_file = os.path.join(save_dir, f'static_animation_theta_{theta_true}.png')
    fig.savefig(out_file)
    plt.close(fig)
    return axs


def plot_animated_theta_posterior(theta_particles_history, theta_true, temp=None, save_dir=''):
    """Animated GIF of the theta posterior evolving step by step.
    theta_particles_history is a per-step DataFrame with a 'posterior' column ({theta: prob})."""
    sns.set_theme()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Theta')
    ax.set_ylabel('Posterior')

    def animate(i):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Theta")
        ax.set_ylabel("Posterior")
        posterior_data = theta_particles_history.iloc[i]['posterior']
        df = pd.DataFrame({'theta': list(posterior_data.keys()),
                           'posterior': list(posterior_data.values())}).sort_values('theta')
        try:
            sns.histplot(data=df, x='theta', weights='posterior', bins=11, kde=True,
                         color='skyblue', alpha=0.4, ax=ax)
        except Exception:
            sns.histplot(data=df, x='theta', weights='posterior', bins=11, kde=False,
                         color='skyblue', alpha=0.4, ax=ax)
        mean_weighted_theta = np.average(df['theta'], weights=df['posterior'])
        predicted_theta = df.loc[df['posterior'].idxmax(), 'theta']
        y_min, y_max = ax.get_ylim()
        ax.vlines(theta_true, y_min, y_max, color='blue', linestyle='-', label='True Theta')
        ax.vlines(mean_weighted_theta, y_min, y_max, color='green', linestyle='--', label='Posterior Mean')
        ax.vlines(predicted_theta, y_min, y_max, color='red', linestyle='-', label='MAP estimate')
        ax.set_title(f"Posterior Distribution at Timestep {i}\n$\\theta$ = {theta_true}")
        ax.legend(loc='upper right')
        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(theta_particles_history),
                                  interval=500, blit=False, repeat=False)
    save_dir = os.path.join(save_dir, 'animation')
    os.makedirs(save_dir, exist_ok=True)
    if temp is not None:
        out_file = os.path.join(save_dir, f'animation_theta_{theta_true}_temp_{temp}.gif')
    else:
        out_file = os.path.join(save_dir, f'animation_theta_{theta_true}.gif')
    ani.save(out_file, writer='pillow', fps=5)
    plt.close(fig)
