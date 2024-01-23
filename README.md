# GameMindsDT
Unleashing the Power of Decision Transformers in Reinforcement Learning for mastering a variety of games and tasks, from classic arcade challenges to complex strategic simulations. Join us in building a custom Transformer model from scratch and exploring new frontiers in AI-driven gameplay.

## Overview
Welcome to **GameMindsDT**, an innovative project at the intersection of Decision Transformers and Reinforcement Learning, aimed at mastering a variety of games and tasks. Our mission is to build a Transformer from scratch, design a robust RL model, and train it to excel in diverse virtual environments.

## Motivation
Decision Transformers represent a paradigm shift in reinforcement learning, offering a more efficient and flexible approach to decision-making. This project endeavors to unlock the full potential of Decision Transformers in various gaming and task-oriented scenarios, pushing the boundaries of what these architectures can achieve.

## Installation
```bash
git clone https://github.com/your-username/GameMindsDT.git
cd GameMindsDT
pip install -r requirements.txt
```
## Features
- **Transformer Building from Scratch**: Deep and personalized learning of Transformer mechanics.
- **Customized RL Model**: Tailoring and fine-tuning models for different types of games and challenges.
- **Rigorous Training and Testing**: Assessment across various games, ranging from classics to contemporary.

## Usage
Guidance on how to use the model with code examples.
```python
# Example code snippet
```
## Contributing
Contributions are welcome! Please read `CONTRIBUTING.md` for details on how you can contribute to our project.

## License
This project is licensed under the [MIT License](LICENSE).

## Team
- Omar Aguilera Vera
- Pol Fernández Blánquez
- Shuang Long Ji Qiu
- Edgar Planell
- Alex Barrachina

## References and aknowledgements
- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- [d3rlpy]https://d3rlpy.readthedocs.io/en/v2.3.0/references/algos.html

## Experiments

We have used docker and notebooks to run the experiments. The notebooks are in the folder `notebooks` and the docker files are in the folder `docker`.

### Docker Experiments

We use docker to run the experiments.

Before running the docker, you need to install the nvidia-docker2 if we want to use the GPU and you don't have installed yet. You need a nvidia GPU at least compatible with CUDA 11.0. You can check the version of CUDA with the command `nvidia-smi`

```bash
sudo apt install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart docker
```

To buid the docker, you need to run the command `docker build -t gamemindsdt:latest .` in the folder `docker`. This command will build the docker with the name `gamemindsdt:latest`. The docker will install all the requirements and will clone the repository in the folder `/home`.

To run the docker, you need to run the command `docker run -it --gpus all --rm -v $(pwd):/home gamemindsdt:latest bash`. This command will run the docker with the name `gamemindsdt:latest` and will mount the current folder in the folder `/home`. The command will open a bash in the docker.

```bash
docker build -t gamemindsdt:latest . 
docker run -it --gpus all --rm -v $(pwd):/home gamemindsdt:latest bash
```

To run the experiments, you need to run the command `python main.py <experiment_name>` in the folder `/home`. The command will run the experiment `<experiment_name>`. For example:
```bash
python main.py test_pendulum
```
```bash
python main.py test_atari_breakout
```


### Pendulum
Perform the test on the simplest gym with d3rl to test the decision transformers. We choose the pendulum environment because it is the simplest and we can see if the DT is able to learn the optimal policy.
**TODO:** Hyperparemter tuning (the simplest possible model).    

### Compare DT with other algorithms
Test the same environments as in the original paper, for the DT, CQL, BC algorithms and compare the results.
Are better DT than CQL and BC?
#### Open AI Gym (HalfCheetah, Hopper, Walker, Reacher)
**TODO:**

#### Atari games (Breakout, QBert, Pong, Seaquest)
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

![Door Key](https://github.com/SwissTonyStark/GameMindsDT/blob/main/assets/rl-video-episode-0.mp4)

And in this chart you can see the comparision of the DT model with the DQN model in terms of solved episodes:

![Chart](https://github.com/SwissTonyStark/GameMindsDT/blob/main/assets/door-key-test-comparision-bar-chart.png)


**TODO:**

### Test DT with different hyperparameters. 
Are important size of model, number of layers, number of heads, number of attention heads, number of epochs, learning rate, batch size, etc.?
Which are the best hyperparameters for DT in every game?
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


## Acknowledgements
TBD.

## Contact
TBD(Contact information for inquiries and collaborations.)


---
![Uploading Logo.png…]()


