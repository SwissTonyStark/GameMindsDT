PyBullet Gym Environments-V0 
For this experiment, we will be using our Decision Transformer in PyBullet Gym Environments-V0 using the library [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet) from [takuseno](https://github.com/takuseno)'s repository. The original idea was to use the latest Mujoco environments from Gymnasium Open Ai, but we didnt' find many datasets available and also we faced several problems installing the library due to deprecated dependencies and license issues.

The library d4rl-pybullet has four replicable enviroments: **Hopper**, **Halfcheetah**, **Ant** and **Walker2D**. For our experiments we used all available enviroments. Each of these enviroments offers diferents types of dataset as can be seen in the following table extracted from d4rl-pybullet repository:

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

Since the dataset offers 3 types of levels: random, medium, and mixed, we need to analyze which of them is the cleanest and optimal to use. Initially, we used the random datasets, but after several trials, we observed that the results were not as expected compared to the other two types. This could have happened because those samples are sampled with a randomly initialized policy, making it hard for the results to converge to an optimal result for the Decision Transformer. Another factor we considered was the amount of samples that the dataset contains. In this type of model, the amount of data impacts the model, especially in the training step, so we discarded datasets that offer fewer samples than the others. Therefore, we decided to work with the medium dataset, and all the results displayed below will be gathered using this dataset.

Before starting training, we analyzed the dataset. The number of samples corresponds to the total steps obtained by adding up the steps of each episode. This is an important point, as we treated the dataset as a sequence of episodes. The preprocessing of the data consisted of the following steps:

  - **Normalization of episodes length:** We excluded episodes with durations shorter than the mean duration. This was done to prevent excessively short episodes, which, during training, would require significant padding, potentially introducing noise into our data
  - **Normalization of Observation Space:** While mean and standard deviation can be derived from the data, we aimed to ensure normalization by applying standard score normalization specifically to the observation space
  - **Generation of Additional Model Inputs:** To accommodate the model's requirements, we manually computed the return-to-go array and timesteps. This was necessary due to the limited information provided by the environment, which includes only the observation space, action space, reward space, and episode terminals

## Decision Transformer modification
The Decision Transformer algorithm used is the same as presented in the official paper. However, for this experiment, some modifications were necessary to adapt to this dataset. The environment we are working with is a continuous space environment. 

This means that the action space will be within the range of [-1, 1], unlike in discrete space environments where the action space represents a probability distribution over the available actions. In continuous environments, actions are typically represented as continuous numerical values, whereas in discrete environments, actions are often indices or labels representing discrete choices.

To accommodate the continuous space, we needed to modify the linearity of the model. Instead of using a softmax function, we employed a hyperbolic tangent.

## Training
To conduct the training, we organized the total samples from the selected dataset into episodes. Consequently, the actual number of episodes became significantly lower than initially anticipated, as the episodes varied in length.
We divided the entire dataset into a training set (80%) and a validation set (20%). Testing will be performed directly using the environment.

## Experiments
### Environment 1: Hopper Pybullet-v0
#### Hypotesis
This environment was the first environment where we trained and tested our Decision Transformer in a continuous space of actions and observations. Since we where starting from scratch with the training of our Decision Transformer, we didn’t want to jump straight forward into an environment of high level of complexity (in dimensionality and computational terms), so the election of the most friendly environment to start, was fundamental for the progress. Obtaining a prominent success in this first environment, will lead us towards the more complex environments and it’s challenges
That’s where Hopper Pybullet comes intoplace. This environment shines for it’s simplicity  in terms of dimensionality in comparison with the other environments available from the same library. 

Here a quick overview of the observations and actions space dimensions:

| observation_space | action_space |
|:-|:-|
|Box(15,)| Box(3,)| 

#### Troubleshooting

Not all stories start off on the right foot, and our initial trials with our Decision Transformer were not going to be an exception. After investing a good amount of time building our early version of the model, and carefully selecting the environment to start training it, problems arose. During the model training process, we experimented with various hyperparameter configurations, and in all of them, the results were quite satisfactory, with both the average loss and validation loss decreasing correctly until they balanced each other out in most configurations. However, when it came to testing the trained models, the situation was a bit more dramatic. 

![replay_decisiontransformer_pybullet_hopper-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/86020737/f15186e4-d4a6-4e9a-b95e-e26404999d25)
![replay_decisiontransformer_pybullet_hopper_4-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/86020737/fe912675-b674-4a4b-bc57-bebbe4b34a81)

As can be seen in the test videos above, the agent consistently performed no or minimal actions throughout all iterations (Video on the left). One of our initial hypotheses was that the issue originated from a potential problem related to reward shaping and exploration-exploitation trade-offs. However, we quickly dismissed this hypothesis when we realized that neither setting the 'return to go' value sufficiently high to motivate exploitation nor fixing the 'return to go' to zero to incentivize exploration resulted in the agent moving. 

This initially puzzled us, until we considered the possibility that the predicted actions by our Decision Transformer may not have been in the correct magnitude order, leading to a lack of expected response from any of the agent’s joints. To address this, we multiplied the predicted actions by a scalar before forwarding  them again to the environment. The test video on the right was recorded during this action magnification, confirming our hypothesis. Subsequent checks in our code revealed that the actions predicted by the Decision Transformer were not adequately scaled during testing, as they were not subject to the same normalization process applied during training. Consequently, the magnitudes of the predicted actions were insufficient to produce a response in the agent's joints when forwarded to the environment.

We made the necessary fixes in the code, and we reran the test. After this changes, our first agent in the Pybullet environment was finally walking. 

#### Hyperparameters
The set of hyperparameters used for our Decision Transformer in the Hopper Pybullet Medimum env-v0:

      "h_dim": 128,  
      "num_heads": 1,
      "num_blocks": 3, 
      "context_len": 20,
      "batch_size": 64,
      "lr": 0.0001,
      "weight_decay": 0.0001,
      "mlp_ratio": 1,
      "dropout": 0.1,
      "train_epochs": 3000,
      "rtg_target": 5000,
      "rtg_scale" :1,
      "constant_retrun_to_go" : True,
      "stochastic_start" : True,
      "num_eval_ep" :10, 
      "max_eval_ep_len":250, 
      "num_test_ep":10,  
      "max_test_ep_len":1000,
      "state_mean" : dataset_observations_mean,
      "state_std" : dataset_observations_std, 
      "render" : False


#### Results
We have conducted several rounds with different agents trained using various hyperparameter setups, and we've found that among all the parameters, slightly increasing the context length led to better performance in this environment. The results of the three runs for the best agents are shown as follow:

![Test Bullet](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/5c032f11-40fe-466a-9a29-1ce77fabd568)

From the test videos performed for those three agents, we can highlight that all three quickly started learning the policy. However, only the last agent followed the optimal policy, ensuring itself to maintain equilibrium persistently throughout all the episodes, making it the most successful in fulfilling the task

https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/95f4d853-056c-454c-9ecb-eb3fcd5c65cf

### Environment 2: Walker2D Pybullet-v0
#### Hypotesis

We choosed this environment as the second task for our Decision Transformer in a continuous space, due we considered the second most feasable task from the environment. Our initial hypotesis, was that we will probably need more computational demanding hyperparameters for this environment, since the environment space dimensionality was almost doubled in comparison with the Hopper environment. We started with the same configuration, what we quickly realized that the model was not able to learn the optimal policy wit that configuration. Twitching the embeddings dimensonality (from 128 to 256), the context length (from 20 to 40) and the batch size (from 64 to 128) was crucial to make it work, since now the agent had more observations and actions features to consider before inference. In the other hand, this was considerable amount of information that will have to be processed, so to ensure a proper assimilation of the environment needs, we slowly increased the number of heads, number of layers until finding the optimal configuration for our Decision Transformer

Here a quick overview of the observations and actions space dimensions:

| observation_space | action_space |
|:-|:-|
|Box(22,)| Box(6,)| 

#### Troubleshooting
As explained in the hypothesis, we managed to find the set of hyperparameters needed for this environment. However, indirectly, the computational cost of the agents with this new environment setup was increasing significantly, and the machines that we were using were starting to reach their limits. Luckily, we managed to find a more powerful machine to continue with this more demanding configuration.

#### Hyperparameters
The set of hyperparameters used for our Decision Transformer in the Walker2D Pybullet Medimum env-v0:

      "h_dim": 256,  
      "num_heads": 8,
      "num_blocks": 4, 
      "context_len": 40,
      "batch_size": 128,
      "lr": 0.001,
      "weight_decay": 0.0001,
      "mlp_ratio": 4,
      "dropout": 0.1,
      "train_epochs": 3000,
      "rtg_target": 5000,
      "rtg_scale" :1,
      "constant_retrun_to_go" : True,
      "stochastic_start" : True,
      "num_eval_ep" :10, 
      "max_eval_ep_len":250, 
      "num_test_ep":10,  
      "max_test_ep_len":1000,
      "state_mean" : dataset_observations_mean,
      "state_std" : dataset_observations_std, 
      "render" : False
     
#### Results
We have conducted several rounds with different agents trained using various hyperparameter setups, and like in Hopper's Pybullet environment, increasing slightly the context length led to a better performance in this environment. Also as expected, increasing the dimensionality of the hidden layers of our embeddings translated into a significant reduction in the training and validation loss, allowing the training to converge quicker. The results of the three best-performing agents are shown below:

![TrainingLoss-Train-Walker2D](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/a2a3e92d-4751-4e32-b597-104393289366)
![ValidationLoss-Train-Walker2D](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/89ae508c-c9b1-4efc-9d00-e368b9957635)

The results on test for the 3 best-performing agents:

![Test Walker2D](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/d9ca989d-a3e2-4af2-84c7-cf19136c8a63)

As the displayed results highlight, while we achieved a better reduction in loss during training with the high-dimensional embedding and context_length agent, the results during testing differ quite significantly from what was expected. We assume that the reason for the significant decrease in loss during training, but not during testing, could be associated with overfitting, perhaps due to the limited amount of data used for training and evaluation. That being said, a video of the final trained agent can be seen as follows:

https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/db4e9855-b892-4103-a222-0720eb026d4a


### Environment 3: Halfcheetah Pybullet-v0
#### Hypotesis
According the logic followed for choosing the first environments, we placed the halfcheetah in the tird position in our to-do list. The dimensionality of the observation and action spaces of this environmant is quite similar to the Walker2D environtment, so we head-up with the hypotesis that a similar set of hyperparameters like the ones used during the fine tunning of the previous environtment will work. 

| observation_space | action_space |
|:-|:-|
|Box(26,)| Box(6,)| 

#### Troubleshooting

As we mentioned, we states as initial hypotesis that a set of hyperparameters very similar to the ones used for fine-tuning the Walker2D would be a good starting point for tackling this new task. However, it came as no surprise when we conducted the first training and testing sessions, only to realize that while the training appeared to converge successfully (at least upon first examination), the agent wasn't moving at all during testing (sound familiar?). Based on the experience we gathered from earlier experiments, we attempted various configurations of hyperparameters until we reached the computational limits of our new machine. After numerous training and testing sessions (up to 100 in this environment alone), we found ourselves feeling quite lost. At this point, we considered the possibility that perhaps the root cause laid down with our decision transformer. We invested a significant amount of time reviewing all the replays from both training and testing phases, attempting to identify what we might have missed after all those tests. It was then that we made the following observations:

![TrainCheckpointN9_halfcheetah_1-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/3e6ca285-19a3-49a0-b33b-5f462405b66d)
![TrainCheckpointN9_halfcheetah_2-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/31ad80ad-1f42-438e-b841-bcd45f4ec96d)
![TrainCheckpointN2_halfcheetah_4-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/4e06216c-b0db-4713-be25-1f942713bd20)


Above, you can see the most common result after training (Video on the left), where the agent was simply falling straight down. In some other trials, we found that the agent had actually learned some of the body movements needed to perform the first step (Video in the middle), but it seemed that during training, the focus was more on standing still than on walking, as can be seen in the last video. 

After conducting some research (the information available was quite limited, especially for this early version of the library), we decided to bet on the possibility that the problem might be related to positional embeddings. Therefore, we implemented Global Positional Embedding for our Decision Transformer and ran trains and tests again. Unfortunately, the results were not better, but worse. To this day, we continue to search for the possible root cause.

#### Hyperparameters
The set of hyperparameters used for our Decision Transformer in the Halfcheetah Pybullet Medimum env-v0:

      "h_dim": 256,  
      "num_heads": 8,
      "num_blocks": 4, 
      "context_len": 40,
      "batch_size": 128,
      "lr": 0.001,
      "weight_decay": 0.0001,
      "mlp_ratio": 4,
      "dropout": 0.1,
      "train_epochs": 3000,
      "rtg_target": 5000,
      "rtg_scale" :1,
      "constant_retrun_to_go" : True,
      "stochastic_start" : True,
      "num_eval_ep" :10, 
      "max_eval_ep_len":250, 
      "num_test_ep":10,  
      "max_test_ep_len":1000,
      "state_mean" : dataset_observations_mean,
      "state_std" : dataset_observations_std, 
      "render" : False
      
#### Results
As crazy as it sounds, we didn't accomplish the task of teaching our Decision Transformer to learn the optimal policy in the Halfcheetah Pybullet-v0 environment. But what if the agent learned to walk backwards?

https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/0ab9369f-c0fd-4daf-8131-18c9abf4f3c2

After spending countless hours trying various combinations of hyperparameters, debugging existing code, and implementing new features to gather more information, our beloved Decision Transformer decided to take a sideways approach and chose to learn a policy that enabled it to walk backwards. Should we consider this a win-win?


### Environment 4: Ant Pybullet-v0
#### Hypotesis
The Ant Pybullet task was by far the most challenging of the tasks available in the Pybullet environment, so we kept it as the last one in order to ensure progress over the rest of them. Unfortunately, the poor results obtained by our agent in the Halfcheetah environment left us with a feeling that the agent would not perform better in the Ant Environment. Even though we attempted several runs with different hyperparameters (based on the feedback gathered from previous environments), but as expected, the results were not the best so we quickly dismissed this environment for this project.

Here a quick overview of the observations and actions space dimensions:

| observation_space | action_space |
|:-|:-|
|Box(28,)| Box(8,)| 

#### Troubleshooting
As described in the Troubleshooting section of the Halfcheetah Pybullet environment, we have not yet found the root cause of why the agent does not generalize properly in these last two environments. For us, this is still an open topic that we will try to follow up on after gathering more testing. Below, we have selected a few of the best highlights from training and test runs for the Ant Environment, where the best attempts to follow the optimal policy can be seen:

![TrainCheckpointN8_1-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/08d377e2-3ee5-4b14-a885-ac3b2eff3408)
![TrainCheckpointN4_2-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/c71d1a80-815d-4a79-b71f-9b667f97452b)
![TrainCheckpointN6_3-ezgif com-speed](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/ef58e7bb-22fd-4d22-b5ff-4f872751e251)


#### Hyperparameters
The set of hyperparameters used for our Decision Transformer in the Ant Pybullet Medimum env-v0:

      "h_dim": 256,  
      "num_heads": 8,
      "num_blocks": 4, 
      "context_len": 40,
      "batch_size": 128,
      "lr": 0.001,
      "weight_decay": 0.0001,
      "mlp_ratio": 4,
      "dropout": 0.1,
      "train_epochs": 3000,
      "rtg_target": 5000,
      "rtg_scale" :1,
      "constant_retrun_to_go" : True,
      "stochastic_start" : True,
      "num_eval_ep" :10, 
      "max_eval_ep_len":250, 
      "num_test_ep":10,  
      "max_test_ep_len":1000,
      "state_mean" : dataset_observations_mean,
      "state_std" : dataset_observations_std, 
      "render" : False

## Installation
You can choose between two ways to install the project: using Docker or creating a Conda environment or something similar. If you prefer to use Docker, at the root of the repository, you will find the dev-container for its installation in Visual Studio. In the same root, the README will guide you through the steps to build the Docker images.

### In Conda
The steps to build the Conda environment are described below:
```
# Create Conda Environment
conda create -n gamemindsDT python=3.10.12
conda activate gamemindsDT
pip install git+https://github.com/takuseno/d4rl-pybullet
pip install -r requirements.txt
```
###

### Considerations and Known Issues
Some users may encounter issues with certain dependencies, particularly with ffmpeg and the H264 codec, in specific environments. Below are some important considerations:

- **ffmpeg and H264 Codec Issue - Ubuntu:** If you encounter issues related to ffmpeg and the H264 codec in your Ubuntu OS, you can find a solution [here](https://stackoverflow.com/questions/70247344/save-video-in-opencv-with-h264-codec).
- **ffmpeg and H264 Codec Issue - Windows:** If you encounter issues related to ffmpeg and the H264 codec in you Windows OS, you can find a solution [here](https://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/).
  
## Run the code
```
# Go into project folder
cd d4rl_pybullet_dt\code

# Run the code
python main.py
```
### User Interface Guide
Once the code is running, a simple user interface will pop up. This simple user interface has been developed to guide the users through the functionalities of the code. There's basically 3 main options to choose:

![UserInterface - MainMenu](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/20d8e1a1-e8d0-4585-9010-ebfd4d72b8df)

### Train a DT in Pybullet-V0 from Scratch
With this option, users can select the desired environment from the available options to start the training a new agent.

![UserInterface - Option1 -EnvSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/a7c1fe66-ff24-4ec8-96ea-dcec1a3145be)

Once the user selects the environment, training begins, and updates and statistics of the training progress are displayed in real-time. Training metrics are not only displayed but also registered in Weights & Biases for further analysis. Videos are generated locally (and logged in Weights & Biases) throughout the training process, allowing users to visualize the training progress of the agent along different checkpoints for the whole training.

### Overview Pretrained Models Config
This utility allows users to review the configuration and hyperparameters of the selected pretrained agent, as well as the environment in which it was trained

![UserInterface - Option2 -PretrainedAgentSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/f9add7a0-4780-4e0a-a48f-85f8c4f80dad)

### Test a Pretrained DT in Pybullet-V0
With the testing option, users can select any of the pretrained agents to evaluate its performance in any of the available environments. The testing process will generate a video at the end, capturing the interactions of the agent with the environment throughout all the test epochs

![UserInterface - Option3 -PretrainedAgentSelected - EnvironmentSelected](https://github.com/SwissTonyStark/GameMindsDT/assets/149005566/fe39bb34-d51a-44cb-8fb1-da8c2c4a904a)
 
## Acknowledgements:

We would like to acknowledge [takuseno](https://github.com/takuseno) the author(s) of the GitHub repository [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet) for providing the dataset and code that served as the foundation for implementing my decision transformer model. Their contributions were essential in the development of this project.

We would like to acknowledge [kzl](https://github.com/kzl)the authors of the paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://github.com/kzl/decision-transformer) for their valuable contributions to the field of reinforcement learning. Their work has inspired and informed our own efforts in developing this project.

Pybullet Official Documentation:
* Used as a general purpouse documentation for troubleshooting with the environment - [PyBullet Quickstart Guide] (https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.btdfuxtf2f72)
* [Reinforcement Learning Gym Envs] (https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.wz5to0x8kqmr)

d3rlpy: 
* A collection of Reinforcement Learning baselines and algorithms for model-based reinforcement learning: We have used his GlobalPositionEncoding. (https://github.com/takuseno/d3rlpy/tree/v2.3.0)
  
hugging_face:
* Used the Hugging Face library to use the GPT-2 model and the Decision Transformer model - [Reinforcment Learning Models - Desicion Transformer](https://huggingface.co/docs/transformers/main/model_doc/decision_transformer)
  
  

