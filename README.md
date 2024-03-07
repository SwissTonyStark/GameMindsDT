# GameMindsDT Project Overview

Welcome to GameMindsDT, where we combine the power of Decision Transformers and Reinforcement Learning to master a myriad of gaming challenges. Our project encapsulates the journey of building a custom Transformer model from the ground up, meticulously tailoring reinforcement learning models, and harnessing them to excel in a diverse array of virtual environments. From classic arcade arenas to sophisticated strategic simulations, GameMindsDT is at the forefront of AI-driven gameplay exploration.

## Table of Contents
- [Project Motivation](#project-motivation)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [License](#license)
- [Project Management](#project-management)
  - [Team Division](#team-division)
  - [Algorithms and Environments](#algorithms-and-environments)
  - [Evolution of the Decision Transformer in our project](#evolution-of-the-decision-transformer)
- [Experiments](#experiments)
- [Introduction to Docker](#introduction-to-docker)
- [Acknowledgements](#acknowledgements)

## Project Motivation
Decision Transformers represent a paradigm shift in reinforcement learning, offering a novel, efficient, and versatile approach to decision-making. Through GameMindsDT, we aim to unlock the untapped potential of these transformers across a broad spectrum of games and tasks, pushing the limits of AI capabilities in virtual environments.

## Installation
```bash
git clone https://github.com/your-username/GameMindsDT.git
cd GameMindsDT
pip install -r requirements.txt
```

## Features
- **Transformer Building from Scratch**: A deep dive into the mechanics of Transformers for a customized learning experience.
- **Customized RL Model**: Tailoring models to suit varied gaming challenges.
- **Rigorous Training and Testing**: Extensive evaluation across a multitude of games, from classic to modern titles.

## Usage
- Comprehensive guide on model utilization with code examples.
- 
## License
This project is licensed under the [MIT License](LICENSE).

## Project Management
### Team Division
- **Omar Aguilera Vera** 
- **[Pol Fernández Blánquez](https://www.linkedin.com/in/polfernandezblanquez/)** 
- **Shuang Long Ji Qiu** 
- **Edgar Planell**
- **Alex Barrachina**

### Algorithms and Environments
**TODO**

### Evolution of the Decision Transformer in our project
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Hierarchical Decision Transformer](https://arxiv.org/abs/2209.10447)
- [Elastic Decision Transformer](https://kristery.github.io/edt/)
- [Critic-Guided Decision Transformer](https://arxiv.org/abs/2312.13716)
- [Online Decision Transformer](https://arxiv.org/pdf/2202.05607.pdf)
- [Constrained Decision Transformer a](https://www.offline-saferl.org/)
- [Constrained Decision Transformer b](https://arxiv.org/abs/2302.07351)

## Experiments
We have used docker and notebooks to run the experiments. The notebooks are in the folder `notebooks` and the docker files are in the folder `docker`.

### Pendulum
Perform the test on the simplest gym with d3rl to test the decision transformers. We choose the pendulum environment because it is the simplest and we can see if the DT is able to learn the optimal policy.
**TODO:** Hyperparemter tuning (the simplest possible model).    

### Compare DT with other algorithms
Test the same environments as in the original paper, for the DT, CQL, BC algorithms and compare the results.
Are better DT than CQL and BC?
#### Open AI Gym (HalfCheetah, Hopper, Walker, Reacher)
**TODO:**

#### Minigrid (Door Key)
This environment has a key that the agent must pick up in order to unlock a goal and then get to the green goal square. This environment is difficult, because of the sparse reward, to solve using classical RL algorithms. It is useful to experiment with curiosity or curriculum learning.

##### Dataset creation and DQN Training
Training this environment with DQN has been quite tough. We've had to add several curriculum learning tricks, extra rewards, etc. You can see the code in 01-train-door-key-16x16-dqn-d3rlpy.ipynb.
This experience has served to demonstrate that environments with long-term rewards are difficult to train with classical model-free RL algorithms, and in this way, we can see the power of DT.
Due to the limitations of the library, we were unable to use sequences such as LSTM to create the DQN model. Without having memory, we had to send the entire world to our model so that it would know whether the door was open or closed. Probably with LSTM, the results would have been better. Despite this, we have been able to train our DT with the dataset generated by our DQN agent and obtain good results.

##### Testing DT
With our trained DQN agent, in the notebook 02-test-door-key-16x16-d3rlpy we can see how to generate a dataset for later training our DT.
We have generated a Dataset by adding entropy to the actions of our DQN agent and progressively reducing it to have a greater variety of data.
Once we obtained the dataset, we trained our DT model, without any kind of trick, on the door-to-key-16x16 environment. That is, the environment becomes much more complicated than the one used for DQN. This is a very interesting result, as we can generate datasets in easy environments, for example, a robot operated by an human, which we then train in DT with the data obtained from a LIDAR.
We tested our DT model and obtained very good results.

##### Results
We evaluate the DT-trained agent by comparing it with the DQN agent. We can see that the DT agent is superior to the DQN agent. The DT agent solves the environment more frequently than the DQN agent. The most important thing to observe is that the DT agent can solve the environment with only partial observation, and without being rewarded for picking up the key or opening the door. The DT agent manages to solve the environment solely with the reward for reaching the goal. Furthermore, the DT agent can solve the environment using only the data generated by the DQN agent. The DT agent is capable of learning from the data produced by the DQN agent. The result is an agent that is better than its 'teacher'. 

Despite this, we have noticed that the training is very unstable, and it can yield very different results in each epoch. We need to continue investigating this issue a little bit more.

Here you can see a video of the DT model playing the door-to-key-16x16 environment:

https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/f37aaa6a-2002-48d0-96f6-75ce65a02d30

And in this chart you can see the comparision of the DT model with the DQN model in terms of solved episodes:

![Chart](https://github.com/SwissTonyStark/GameMindsDT/blob/main/assets/door-key-test-comparision-bar-chart.png)

##### Comparing DT with other offline algorithms in Door Key 16x16 environment

We will make new experiments in Door Key 16x16 environment from Minigrid Gym with Offline algorithms from d3rlpy. You can see the code in notebook 03-test-door-key-16x16-offline-algorithms-d3rlpy.ipynb.

We test the following offline algorithms:
* Discrete Decision Transformer (DT)
* Discrete Behaviour Clonig (BC)
* Discrete CQL (CQL)

We have tried to make a fair comparison, using the same dataset and without tweaking too many hyperparameters.

The intuition behind this is that in problems of sparse reward or delayed reward, offline RL algorithms can be very effective because they learn from experts. And they don't need as much exploration.

As we can see, the best algorithm by far is DT.

![Chart](https://github.com/SwissTonyStark/GameMindsDT/blob/main/assets/door-key-test-offline-comparision-bar-chart.png)

#### Atari games (QBert, Seaquest, Pong, Breakout)

We have trained the four games from the original paper using the Decision Transformer from d3rlpy. Here we show example videos of a game. The source code used for the training is in the Experiments section / Docker.

https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/adf01b3d-a509-4941-ae85-4cdf5f98e090

https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/56462ebf-ef7c-40fa-ab68-2d1576b308c4 

https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/07b4d3c3-8e25-4e06-9a88-a88c9b2c5632

https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/25c57515-57ee-4cac-a0a3-3f70b3b7251a

##### Comparing DT with other offline algorithms in Atari games (QBert, Seaquest, Pong, Breakout)

We have trained the 4 games with the d3rply library on the 3 algorithms, DT, BC, and CQL, without hardly touching the hyperparameters, and the results show that DT is better compared to the others.
For the test, we have run 100 matches of each game for each algorithm. And we have averaged the results.
![atari_results](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/dd963e67-5b0e-45c2-b26d-578ec7ff2ea6)

### Test DT with different hyperparameters. 
Testing Decision Transformers (DT) with different hyperparameters and data experiences across various games and environments is a complex but crucial aspect of enhancing reinforcement learning (RL) models. The literature and experiments indicate that factors such as model size, number of layers, learning rate, batch size, and the mix of expert vs. non-expert training data significantly influence the performance of RL agents, including DTs.

*Hyperparameters* like the learning rate, batch size, number of layers, and model size are pivotal for the training efficiency and final performance of DT models. For example, the learning rate controls the step size at each iteration while moving toward a minimum of a loss function, impacting the convergence speed and stability of the learning process. Similarly, the batch size influences the model's ability to generalize from the training data, while the number of layers and model size can affect the model's capacity to learn complex patterns and behaviors.

In reinforcement learning, especially with Decision Transformers acrossing specific Atari Games, optimizing these hyperparameters is essential for achieving high performance across different games and environments. 

#### Open AI Gym (HalfCheetah, Hopper, Walker, Reacher)
**TODO:**

#### Atari games (Breakout, QBert, Pong, Seaquest)
**TODO:**

#### Minigrid (Door Key)
**TODO:**

### Test DT with different data experiences.
Are important the size of the dataset and the type of data?
#### Atari games (mixed, expert)
**TODO:**

#### Open AI Gym (medium, medium-replay, medium-expert)
**TODO:**

## Introduction to Docker

Docker is an open-source containerization platform that enables developers to package applications into containers—standardized executable components combining application source code with the operating system (OS) libraries and dependencies required to run that code in any environment.

### Why Docker?

- **Isolation**: Docker ensures that your application works in a consistent and isolated environment by packaging it along with its environment.
- **Resource Efficiency**: Containers share the host OS kernel, are much lighter weight than virtual machines, and start up quickly.
- **Simplified Development**: Avoid the "it works on my machine" problem by packaging the application with its environment.
- **CI/CD Integration**: Docker integrates with continuous integration and deployment workflows, allowing for automated testing and deployment.

### Prerequisites
Before getting started, make sure you have Docker installed on your machine. You can download and install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).

### Usage Instructions

#### Installation and Initial Setup
1. **Installing Docker Desktop:**
   - Download and install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).
   - Ensure Docker Desktop is running and the Docker Daemon is active.

#### Building the Base Image
The base image includes PyTorch, CUDA, and other necessary dependencies. Navigate to the root directory of your repository.

- Navigate to the root directory of your repository.
- Run the following command:
```
cd GameMindsDT/
docker build -t nvidia-pytorch:base .
```
Then will you have installed the base image to run our project in a container! But keep reading, you will need to add some extensions,
new images which contain more dependencies.

#### Building the MineRL Image
The MineRL image is required if you wish to work with the MineRL experiment. Navigate to the directory containing the Dockerfile for MineRL.

- Navigate to the directory containing the Dockerfile for MineRL.
- Run the following command:
```
cd GameMindsDT/dt-mine-rl-project
docker build -t minerl-dt .
```
#### Building the PyBullet Image
The PyBullet image is required if you wish to work with experiments that require PyBullet. Navigate to the directory containing the Dockerfile for PyBullet.

- Navigate to the directory containing the Dockerfile for PyBullet.
- Run the following command:
```
cd GameMindsDT/d4rl_pybullet_dt
docker build -t pybullet-dt .
```
#### Running a Container
Once you have built the image you need, you can run a container based on that image. Make sure to replace `nvidia-pytorch:base`, `minerl-dt`, or `pybullet-dt` with the name of the image you have built.
You will find your images in Docker Desktop, it's recommendable to check if you have them all, but you won't need to run them manually. Visual Studio Code does it for you, it runs an instance of them.

#### Setting Up Development Environment in VSCode
After building the images, open the project in Visual Studio Code. Use the "Reopen in Container" feature from the top menu. Select the desired container from the available options.

The Docker containers are defined in the `./devcontainer` directory, where you can find their configurations.

That's it! You are now ready to work with Docker and use the built images for your projects with MineRL and PyBullet.

## References and Acknowledgements
- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- [d3rlpy](https://d3rlpy.readthedocs.io/en/v2.3.0/)
- [Hierarchical Decision Transformer for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2209.10447)
- [Elastic Decision Transformer for Generalizable Offline Reinforcement Learning](https://kristery.github.io/edt/)
- [Critic-Guided Decision Transformer for Offline Hindsight Information Matching](https://arxiv.org/abs/2312.13716)
- [Online Decision Transformer](https://arxiv.org/pdf/2202.05607.pdf)
- [Constrained Decision Transformer for Safe Reinforcement Learning](https://www.offline-saferl.org/)
- [Docker: Containerization Platform](https://www.docker.com/)
- [OpenAI Gym: A Toolkit for Developing and Comparing Reinforcement Learning Algorithms](https://gym.openai.com/)
- [Minigrid: A Minimalistic Gridworld Environment for OpenAI Gym](https://github.com/maximecb/gym-minigrid)


2023-2024
