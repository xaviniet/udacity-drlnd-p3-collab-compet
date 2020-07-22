from unityagents import UnityEnvironment
from ddpg_agent import Agent
import numpy as np
import torch

ENV_FILE = './env/Tennis.app'
ACTOR_WEIGHTS = './checkpoint_actor_best.pth'
CRITIC_WEIGHTS = './checkpoint_critic_best.pth'


def play(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations



    agent = Agent(state_size=states.shape[1], action_size=action_size, random_seed=2)

    agent.reset()

    agent.actor_local.load_state_dict(torch.load(ACTOR_WEIGHTS))
    agent.critic_local.load_state_dict(torch.load(CRITIC_WEIGHTS))
    scores = []
    score = np.zeros((2,))
    while True:
        agent.reset()   
        actions = agent.act(states)

        env_info = env.step(actions)[brain_name]
        states = env_info.vector_observations
        score += np.array(env_info.rewards)
        dones = env_info.local_done
        if np.sum(dones) > 0:
            break

    print('Scores: {}'.format(score))
    
    

if __name__=='__main__':
    env = UnityEnvironment(file_name=ENV_FILE)
    
    num_games = 5
    for i in range(num_games):
        play(env)
    
    env.close()

