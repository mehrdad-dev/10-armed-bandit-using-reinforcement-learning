import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


PLAYS = 1000
REPETITION = 2000
EPSILONS = [0, 0.01, 0.1]


class N_Armed_Bandit:
    def __init__(self, arms: int, plays: int, epsilon: float) -> None:
        self.arms = arms
        self.plays = plays
        self.epsilon = epsilon
        self.real_action_values = np.random.normal(
            loc=0, scale=1, size=self.arms)
        self.rewards_history = {i: {'sum': 0, 'count': 0}
                                for i in range(self.arms)}
        self.actions_rewards = np.zeros(self.arms)
        self.all_rewards_store = np.zeros(self.plays)
        self.optimal_action_store = np.zeros(plays)


    def e_greedy_selection(self, step: int) -> int:
        if np.random.rand() > self.epsilon:
            index = np.argmax(self.actions_rewards)
        else:
            index = np.random.choice(self.arms)

        self.optimal_action_store[step] = (
            index == np.argmax(self.real_action_values))

        return index


    def generate_reward(self, selected_action_index: int) -> float:
        return np.random.normal(loc=self.real_action_values[selected_action_index], scale=1)


    def update(self, reward: float, selected_action_index: int, step: int) -> None:
        self.rewards_history[selected_action_index]['sum'] += reward
        self.rewards_history[selected_action_index]['count'] += 1

        self.actions_rewards[selected_action_index] = (
            self.rewards_history[selected_action_index]['sum'] /
            self.rewards_history[selected_action_index]['count']
        )

        self.all_rewards_store[step] = reward


    def run(self) -> tuple:
        for step in range(self.plays):
            selection_index = self.e_greedy_selection(step)
            reward = self.generate_reward(selection_index)
            self.update(reward, selection_index, step)

        return self.all_rewards_store, self.optimal_action_store


if __name__ == '__main__':
    final_results = []
    optimal_action_percentages_list = []

    for epsilon in EPSILONS:
        temp_rewards_results = []
        temp_optimal_actions_results = []

        for step in tqdm(range(REPETITION), desc=f'Epsilon = {epsilon}'):
            rewards, optimal_actions = N_Armed_Bandit(
                arms=10, plays=PLAYS, epsilon=epsilon).run()
            temp_rewards_results.append(rewards)
            temp_optimal_actions_results.append(optimal_actions)

        final_results.append(np.mean(temp_rewards_results, axis=0))
        optimal_action_percentages_list.append(
            np.mean(temp_optimal_actions_results, axis=0))

    x_axis = np.arange(PLAYS)
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, final_results[0], label='e=0.0')
    plt.plot(x_axis, final_results[1], label='e=0.01')
    plt.plot(x_axis, final_results[2], label='e=0.1')
    plt.ylabel('Average Rewards', fontsize=14, labelpad=15)
    plt.xlabel('Plays', fontsize=14, labelpad=15)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2, 1, 2)
    plt.plot(x_axis, optimal_action_percentages_list[0], label='e=0.0')
    plt.plot(x_axis, optimal_action_percentages_list[1], label='e=0.01')
    plt.plot(x_axis, optimal_action_percentages_list[2], label='e=0.1')
    plt.ylabel('% Optimal Action', fontsize=14, labelpad=15)
    plt.xlabel('Plays', fontsize=14, labelpad=15)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()
