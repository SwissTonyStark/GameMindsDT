# Experiment: PyBullet enviroments
In this experiment we are using our Decision Trasnformer in PyBullet enviroments using the library d4rl-pybullet from this repository. The original idea was to use Mujoco enviroment from gymnasium, but since we had problems on installing the library due to deprecated dependencies and lisence problems, we seek another similar enviroment.

The library d4rl-pybullet has four replicable enviroments: **Hopper**, **Halfcheetah**, **Ant** and **Walker2D**. For our experiments we used all available enviroments. Each of these enviroments offers diferents types of dataset as seen in the following table extracted from d4rl-pybullet repository:

- `random` denotes datasets sampled with a randomly initialized policy.
- `medium` denotes datasets sampled with a medium-level policy.
- `mixed` denotes datasets collected during policy training.

| id | task | mean reward | std reward | max reward | min reward | samples |
|:-|:-:|:-|:-|:-|:-|:-|
| hopper-bullet-random-v0 | HopperBulletEnv-v0 | 18.64 | 3.04 | 53.21 | -8.58 | 1000000 |
| hopper-bullet-medium-v0 | HopperBulletEnv-v0 | 1078.36 | 325.52 | 1238.9569 | 220.23 | 1000000 |
| hopper-bullet-mixed-v0 | HopperBulletEnv-v0 | 139.08 | 147.62 | 1019.94 | 9.15 | 59345 |
| halfcheetah-bullet-random-v0 | HalfCheetahBulletEnv-v0 | -1304.49 | 99.30 | -945.29 | -1518.58 | 1000000 |
| halfcheetah-bullet-medium-v0 | HalfCheetahBulletEnv-v0 | 787.35 | 104.31 | 844.91 | -522.57 | 1000000 |
| halfcheetah-bullet-mixed-v0 | HalfCheetahBulletEnv-v0 | 453.12 | 498.19 | 801.02 | -1428.22 | 178178 |
| ant-bullet-random-v0 | AntBulletEnv-v0 | 10.35 | 0.31 | 13.04 | 9.82 | 1000000 |
| ant-bullet-medium-v0 | AntBulletEnv-v0 | 570.80 | 104.82 | 816.79 | 70.87 | 1000000 |
| ant-bullet-mixed-v0 | AntBulletEnv-v0 | 255.40 | 196.22 | 609.66 | -32.74 | 53572 |
| walker2d-bullet-random-v0 | Walker2DBulletEnv-v0 | 14.98 | 2.94 | 66.90 | 5.73 | 1000000 |
| walker2d-bullet-medium-v0 | Walker2DBulletEnv-v0 | 1106.68 | 417.79 | 1394.38 | 16.00 | 1000000 |
| walker2d-bullet-mixed-v0 | Walker2DBulletEnv-v0 | 181.51 | 277.71 | 1363.94 | 9.45 | 89772 |

## Dataset
Since the dataset offers 3 types of levels: `random`, `medium` and `mixed`, we need to analise which of them are the cleanest and optim one to use. First we used the `random` datasets, but after different trials we saw that the results was not as we expected compared to the other two types. This could be happend because those samples are sampled wiht a randomly initialized policy, making the results hard to converge to a optim result for the Decision Transformer. Another artifact we took care was the amount of samples that the dataset has. In this kind of models, the amount of data impacts on the model, specially in the training step, so due to that, we discarded thoses datasets that offers less samples than the other ones. So the decision was to work the the `medium` dataset.

Before start training, we analised the dataset. The number of samples corresponds to the total steps obtained from adding up the steps of each episode. This is an important point, since we trated the dataset as a sequence of episodes.
The preproecess of the data consisted on:
  - Check the number of steps an episode has. We removed those when the duration was less than the mean duration. The main reason is to void episode too shorts, because at training, those episoded will have a large amount of padding, thus adding noise to our data.
  - Although the mean and the standard deviation can be extracted from the data, we wanted to ensure that it is normalized, so we jsut applied a standard score normalization to the observation space. 

Additional data were needed to feed into the model. We have had to calcultae the return-to-go array and the timesteps manually since the enviroment only provides the observation space, action space, reward space and episode terminals.

## Decision Transformer modification
The Decision Transformer algorithm used is the same presented in the official paper, but for this experiment, some changes must be done to work wiht this dataset correctly. Sampling the actions space in order to gather knowledgement of the enviroment, we realised that the values of the actions can take any value within the range (-1,1), meaning that we are working wiht a continuous actions space. In order to work with continous space, we needed to change the linearlity of the model, instead of using a softmax we used a hyperbolic tangent. 

## Training
To do the training, we divided the samples into episoded, so at the end the number of samples of the data sets corresponds to the number of episodes. Due to this, the actual number of examples is much less than the initial one since it is obvious that the number of episodes << number of samples.
We splitted the data into a training set (80%) and a validation set (20%). The test is going to be done directly using the eviroment.

## Experiments
### Environment 1: Hopper Pybullet-v0
#### Hypotesis
This environment was the first environment where we trained and tested our Decision Transformer in a continuous space of actions and observations. Since we where starting from scratch with the training of our Decision Transformer, we didn’t want to jump straight forward into an environment of high level of complexity (in dimensionality and computational terms), so the election of the most friendly environment to start, was fundamental for the progress. Obtaining a prominent success in this first environment, will lead us towards the more complex environments and it’s challenges
That’s where Hopper Pybullet comes intoplace. This environment shines for it’s simplicity  in terms of dimensionality in comparison with the other environments available from the same library. Here a quick overview of the actions and observations space dimensions:

#### Troubleshooting

Not all stories start off on the right foot, and our initial trials with our Decision Transformer were not going to be an exception. After investing a good amount of time building our early version of the model, and carefully selecting the environment to start training it, problems arose. During the model training process, we experimented with various hyperparameter configurations, and in all of them, the results were quite satisfactory, with both the average loss and validation loss decreasing correctly until they balanced each other out in most configurations. However, when it came to testing the trained models, the situation was a bit more dramatic. 

![replay_decisiontransformer_pybullet_hopper-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/86020737/f15186e4-d4a6-4e9a-b95e-e26404999d25)
![replay_decisiontransformer_pybullet_hopper_4-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/86020737/fe912675-b674-4a4b-bc57-bebbe4b34a81)

As can be seen in the test videos above, the agent consistently performed no or minimal actions throughout all iterations (Video on the left). One of our initial hypotheses was that the issue originated from a potential problem related to reward shaping and exploration-exploitation trade-offs. However, we quickly dismissed this hypothesis when we realized that neither setting the 'return to go' value sufficiently high to motivate exploitation nor fixing the 'return to go' to zero to incentivize exploration resulted in the agent moving. 

This initially puzzled us, until we considered the possibility that the predicted actions by our Decision Transformer may not have been in the correct magnitude order, leading to a lack of expected response from any of the agent’s joints. To address this, we multiplied the predicted actions by a scalar before forwarding  them again to the environment. The test video on the right was recorded during this action magnification, confirming our hypothesis. Subsequent checks in our code revealed that the actions predicted by the Decision Transformer were not adequately scaled during testing, as they were not subject to the same normalization process applied during training. Consequently, the magnitudes of the predicted actions were insufficient to produce a response in the agent's joints when forwarded to the environment.

We made the necessary fixes in the code, and we reran the test. After this changes, our first agent in the Pybullet environment was finally walking. 


### Results
We have conducted several rounds with different agents trained using various hyperparameter setups, and we've found that among all the parameters, slight changes in the context length led to better performance in this environment. The results of the three runs for the best agents are shown as follow:

![Test Bullet](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/5c032f11-40fe-466a-9a29-1ce77fabd568)

From the test videos performed for those three agents, we can highlight that all three quickly started learning the policy. However, only the last agent followed the optimal policy, ensuring itself to maintain equilibrium persistently throughout all the episodes, making it the most successful in fulfilling the task

https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/95f4d853-056c-454c-9ecb-eb3fcd5c65cf

### Environment 2: Walker2D Pybullet-v0
#### Hypotesis
#### Troubleshooting
#### Results

### Environment 3: Halfcheetah Pybullet-v0
#### Hypotesis
#### Troubleshooting
#### Results

### Environment 4: Ant Pybullet-v0
#### Hypotesis
#### Troubleshooting
#### Results

## Installation
You can choose between two ways to install the project: using Docker or creating a Conda environment or something similar. If you prefer to use Docker, at the root of the repository, you will find the dev-container for its installation in Visual Studio. In the same root, the README will guide you through the steps to build the Docker images.

The steps to build the Conda environment are described below:
### In Conda
#### Go into folder 
cd d4rl_pybullet_dt\code

#### Create Conda Environment
conda create -n gamemindsDT python=3.10.12
conda activate gamemindsDT
pip install git+https://github.com/takuseno/d4rl-pybullet
pip install -r requirements.txt

### Usage
A simple user interface has been developed to guide users through the functionalities of the code

![UserInterface - MainMenu](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/20d8e1a1-e8d0-4585-9010-ebfd4d72b8df)

### Train a DT in Pybullet-V0 from Scratch
With this option, users can select their desired environment from the available options to start the training.

![UserInterface - Option1](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/04843564-1ff9-4f28-9500-69b04f0ef09a)

Once the user selects the environment, the training begins, and updates and stats of the training progress are displayed in real-time. Training metrics are not only displayed but also registered in Weights&Biases for further analysis. Videos are generated locally (and logged in Weights&Biases) throughout the training process, allowing users to visualize the training progress of the agent.

![UserInterface - Option1 -EnvSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/a7c1fe66-ff24-4ec8-96ea-dcec1a3145be)

### Overview Pretrained Models Config
This utility allows users to review the environment and configuration used for a certain pretrained agent.

![UserInterface - Option2](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/6d6e25d3-a21a-4987-a9e3-e75d78f8e634)
![UserInterface - Option2 -PretrainedAgentSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/f9add7a0-4780-4e0a-a48f-85f8c4f80dad)

### Test a Pretrained DT in Pybullet-V0
Finally, with this last option, users can test any of the pretrained agents in any of the available environments. A video will be generated at the end, framing all the epochs the agent has been through.
![UserInterface - Option3 -PretrainedAgentSelected - EnvironmentSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/3ca722a4-a73c-4a88-b868-ed9d09ca1579)
 
## Conclusions


