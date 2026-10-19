import imageio
import matplotlib.pyplot as plt

def save_gif(frames, filepath, fps=4):
    """Saves array of frames as an animated GIF[cite: 1, 2]."""
    imageio.mimsave(filepath, frames, fps=fps)

def plot_training_results(rewards, steps, filepath="output/training_curves.png"):
    """Plots and saves episode rewards and step counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(rewards, color='b')
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    
    ax2.plot(steps, color='r')
    ax2.set_title("Steps Taken per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()