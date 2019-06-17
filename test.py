import sys
import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
def write_test_to_file(file_output, data):
    rewards_labels = ["episode", "rewards"]
    with open(file_output, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        if sum(1 for line in open(file_output)) == 0:
            writer.writerow(rewards_labels)
        for row in data:
            writer.writerow(row)


def test(episode_checkpoint, task, agent, prefix):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state, noise=False) 
        next_state, splitted_rewards, done = task.step(action)
        net_reward = splitted_rewards[-1]
        agent.step(action, net_reward, next_state, done, i_episode, learn=False)
        state = next_state
        if done:
            write_test_to_file(f'data/{prefix}_test.txt', [[int(episode_checkpoint), round(task.score, 2)]])
            break
        
def plot_tests(prefix):
    results = pd.read_csv(f'data/{prefix}_test.txt')
    plt.title(f'Results for {filename}')
    plt.plot(results['episode'], results['rewards'])
    plt.legend()
    plt.show()