import numpy as np


def train_agent(agent, states, merged_data, max_episodes=1000, convergence_threshold=0.001, consecutive_episodes=10):
    training_rewards = []
    consecutive_count = 0
    episodes_to_convergence = None

    for episode in range(max_episodes):
        prev_q_table = agent.get_q_table().copy()
        episode_reward = 0

        for i in range(len(states) - 1):
            state = states[i]
            action = merged_data.loc[hash(tuple(state)) % len(states), 'SubLine_Sent']
            next_state = states[i + 1]
            reward = merged_data.loc[hash(tuple(state)) % len(states), 'Reward']
            episode_reward += reward
            agent.train(state, action, reward, next_state)

        training_rewards.append(episode_reward)

        current_q_table = agent.get_q_table()
        q_diff = np.abs(current_q_table - prev_q_table)
        max_diff = np.max(q_diff)

        if max_diff < convergence_threshold:
            consecutive_count += 1
            if episodes_to_convergence is None:
                episodes_to_convergence = episode + 1
        else:
            consecutive_count = 0

        if consecutive_count >= consecutive_episodes:
            print("Convergence reached. Stopping training.")
            break

    return training_rewards, episodes_to_convergence
