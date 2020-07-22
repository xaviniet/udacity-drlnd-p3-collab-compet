from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

import torch

from collections import deque
from ddpg_agent import Agent

def ddpg( n_episodes=500, max_t=200, train_mode=True):
    env = UnityEnvironment(file_name='./env/Tennis.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=train_mode)[brain_name]

    states = env_info.vector_observations

    agent = Agent(state_size=states.shape[1], action_size=action_size, random_seed=2)

 
    brain_name = env.brain_names[0]
    scores = []
    scores_deque = deque(maxlen=100)
    scores_mean = []
    max_score = -np.Inf
    

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        num_agents = len(env_info.agents)
#         agent.reset()
        score = np.zeros((2,))
        states = env_info.vector_observations
        for t in range(max_t):
            agent.reset()
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states= env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += env_info.rewards
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores_mean.append(np.mean(scores_deque))
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {}'.format(i_episode, np.mean(scores_deque), score), end="")
        
        if np.max(score)> max_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_best.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_best.pth') 
            print('\rSaving Weights for max score old: {} -> new: {} '.format(max_score, np.max(score)))
            max_score = np.max(score)


        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {}'.format(i_episode, np.mean(scores_deque), score), end="")
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        
#         if np.mean(scores_deque)>=0.5:
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
#             torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
#             torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
#             break

    env.close()
    return scores, scores_mean


if __name__=='__main__':
    scores, sc_mean = ddpg(n_episodes=3500, max_t=500)
    
    scores = np.max(np.array(scores),axis=1)
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(sc_mean)+1), sc_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('train_scores.png')
    plt.show()