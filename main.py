import os
import numpy as np
from rescue_env import RescueRobotEnv
from agent import QLearningAgent
from utils import save_gif, plot_training_results

def main():
    # Directory setup
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Environment setup
    env = RescueRobotEnv()
    state_shape = env.observation_space.nvec
    action_size = env.action_space.n

    # Agent setup
    agent = QLearningAgent(state_shape=state_shape, action_size=action_size, lr=0.1, gamma=0.95)

    num_episodes = 800
    rewards_history = []
    steps_history = []

    print("Training Started...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {np.mean(rewards_history[-50:]):.2f} | Epsilon: {agent.epsilon:.2f}")

    # Model save[cite: 1, 2]
    model_path = "model/q_table.npy"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot results
    plot_training_results(rewards_history, steps_history)

    # Evaluation and Render GIF[cite: 1, 2]
    print("Generating evaluation GIF...")
    agent.epsilon = 0.0  # Pure exploitation mode
    state, _ = env.reset()
    frames = [env.render()]
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

    gif_path = "output/rescue_agent.gif"
    save_gif(frames, gif_path)
    print(f"Saved evaluation GIF to {gif_path}")

if __name__ == "__main__":
    main()