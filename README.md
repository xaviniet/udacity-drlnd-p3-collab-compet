# Project 3: Collaboration and Competition

### Introduction

For this project, we train an agent for solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from Unity. 

We use the same solution that we used to solve the  [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, showing the capacity of these algorithms to adapt to and learn different tasks. 

In the environment, we have two agents that control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**. If an agent lets the ball hit the gound or hits it out of bounds, it receives a reward of **-0.01**. Thus, the goal of each agent is to keep the ball in play. 

The observation space has 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Thus, we will have to observations at each step. In this environment the agent have an action space encoded in a vector of **2** continuous values, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of **+0.5** over 100 consecutive episodes. Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get the score for each agent. We take the maximum of these 2 scores. The environment is considered solved when the average of this scores is, at least, +0.5.

### Gettin Started

The version we solve is different from the one in the [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)

1. Download the environment from one of the links below. You only need select the environment that matches your operating system [1]:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

[1]: In the repository we have the Mac OSX versions of the environment

2. Place the file in the repository `p3_collab-compet/` folder, and unzip the file

### Instructions
Follow the instructions in [`Tennis.ipynb`](./tennis.ipynb) to get started with training your own agent.

We have some files and folders to consider:
- `ddpg_agent.py` - definition of the agent and support classes (replay buffer and Ornstein-Uhlenbeck process)
- `train_agent.py` - script for training the agent 
- `Tennis.ipynb` - Here we make experiments and we also can train the agent
- `model` - folder containig trained weights for the model

#### Tennis.ipynb

The notebook where we train the agents (the script `train_agent.py` is almost the same)
- We import the needed modules
- We define the ddpg function where:
    - We initialize the environment
    ```python
    env = UnityEnvironment(file_name=ENV_FILE)
    ```
    - We initialize the agent
    ```python
    agent = Agent(state_size=states.shape[1], action_size=action_size, random_seed=2)
    ```
    
    - We train the agent for *i* episodes
    ```python
    for i_episode in range(1, n_episodes+1):
    ```
- We train the agent calling the function and observe the results

#### model.py
In this file we define the `Actor` and `Critic` classes that contain the Neural Network definition of each one.

#### ddpg_agent.py
We define the `Agent` class and the methods for interacting (`act`, `step`) and training (`learn`, `soft_update`). We also define the `ReplayBuffer` class, needed by the agent on the training of the algorithms coded and `OUNoise` class needed for adding noise with Ornstein-Uhlenbeck process on the `act` method. 

#### model folder
The folder contains two files with the best weights for the Actor and Critic networks that the agent learnt (these are `checkpoint_actor_best.pth` and `checkpoint_critic_best.pth`), and two files with the last weights recorded just when the environment was solved.


    