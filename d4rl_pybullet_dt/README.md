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
### Scenario 1: Hopper

### Scenario 2: Walker2D

### Scenario 3: Halfcheetah

### Scenario 4: Ant

## Conclusions


