import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import Agent
import pandas as pd


# get the default brain
env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

eps_start = 1
eps_end = 0.01
eps_decay = 0.9975
n_episodes = 3000
scores_window = deque(maxlen=100)                           # last 100 scores
episode_data = []                                                 # list containing scores from each episode
scores = []  # stores all the scores for plotting
average_losses = []

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    state = env_info.vector_observations[0]                 # get the current state
    score = 0                                               # initialize the score
    eps = eps_start                                         # initialize epsilon
    action_idx = 0
    while True:
        action_idx += 1
        action = agent.act(state, eps=eps)             # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    eps = max(eps_end, eps_decay*eps)                       # decrease epsilon
    scores_window.append(score)                             # save most recent score
    scores.append(score)                                    # save most recent score
    episode_data.append({
        "episode": i_episode,
        "score": score,
        "mean_scores_window": np.mean(scores_window),
        "state": state,
        "action": action,
        "nn": agent.qnetwork_local.state_dict(),
        "nn_values": agent.nn_values,
        "loss": agent.last_loss
    })                                    # save most recent score
    agent.last_loss = []
    losses = [np.mean(e["loss"]) for e in episode_data]
    average_losses.append(sum(losses) / len(losses))
    if i_episode % 10 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        print('\tLast loss: {}'.format([np.mean(e["loss"]) for e in episode_data]))
        try:
            fig, axes = plt.subplots(2)
            fig.suptitle(f"Episode {i_episode}")
            axes[0].plot(pd.Series([e["mean_scores_window"] for e in episode_data]), 'g-')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Score', color='g')
            axes[0].grid()
            axes[1].plot(pd.Series(average_losses), 'b-')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Loss', color='b')
            axes[1].grid()
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"figures/episode_{i_episode}.png")
        except:
            pass
    if np.mean(scores_window) >= 13.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

        break



