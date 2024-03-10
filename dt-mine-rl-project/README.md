# Decision Transformer for MineRL

We have utilized the [basalt benchmark](https://github.com/minerllabs/basalt-benchmark) framework to evaluate decision transformers. Initially, we began with the Hugging Face Decision Transformer, subsequently transitioning to our own custom implementation. To date, the 'Find Cave' environment is the sole completed setting, where the agent successfully locates a cave roughly 50% of the time. Furthermore, the agent exhibits impressive navigational skills, extricating itself from complex situations and skirting obstacles. For additional environments, we posit the necessity of a Hierarchical Decision Transformer.

## Motivation - From Theory to Practice
Our prior experiments have demonstrated that Decision Transformers are capable of resolving games and standard benchmarks such as Mujoco, Atari, and Minigrid. Nonetheless, their applicability in complex, real-world scenarios remains a question. Due to a lack of extensive data, we elected to utilize Minecraft—a game that, while still being a game, presents complexities far surpassing previous environments. Our objective is to ascertain whether the Decision Transformer can glean insights from human video demonstrations.

![From Theory to Practice](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/2246b179-2b43-4f8e-9995-0142e7a196af)

## Hypothesis
We postulate that the Decision Transformer model should exhibit performance on par or superior to that of behavioral cloning (as leveraged by OpenAI for VPT pre-training). We anticipate its capability to make effective decisions in environments characterized by long-term rewards, akin to the performance in the 'Key-to-Door' Minigrid environment.

## Decision Transformer Trained Exclusively with Videos
This model formulates decisions by analyzing features extracted from frames of human gameplay videos, with the aid of the VPT library. Hence, it learns Minecraft navigation and cave discovery solely through video observation. For targeted learning, videos are trimmed to emphasize the final frames, facilitating cave-finding training.

No complex techniques were employed, barring the deactivation of the inventory button and video trimming. An arbitrary reward was appended at each episode's conclusion without additional contextual information.

## How It Works
Initially, embeddings and associated actions are extracted from the videos utilizing the pre-trained VPT model. Subsequently, our Decision Transformer agent is trained with this data to predict the next action. During testing, embeddings are extracted in real-time using VPT, which the Decision Transformer then utilizes for action prediction, akin to a conventional RL model.

### VPT

OpenAI's Video PreTraining (VPT) is a semi-supervised methodology that employs a modest dataset of videos and corresponding actions to train an inverse dynamics model (IDM). This IDM, more efficient than traditional behavioral cloning, predicts actions by considering both past and future frames. The IDM enables labeling a broader video dataset for behavioral cloning, significantly minimizing the need for direct supervision.

![VPT Schema](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/dabd6445-2d82-4a65-94df-8969acd390d2)

### Training

![Training DT](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/4983723a-06de-4a34-a3d5-829b54067e90)

### Playing

![Playing DT](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/ad7ddafc-f320-4c27-a664-e339c0b7d68c)

*Credits for some images: [OpenAI VPT Paper](https://cdn.openai.com/vpt/Paper.pdf), [Arxiv Paper](https://arxiv.org/pdf/2106.01345.pdf), [Minecraft](https://www.minecraft.net/)*

## Main Challenges and Decisions

Below is a compilation of pivotal decisions shaping our training and testing framework:
 

### Custom DataCollator
Transforms sequences of episodes, embeddings, and actions into structured state, action, and reward-to-go triads for accurate input into the Decision Transformer.

### Action Logits
Implemented a Multicategorical distribution class for three types of action logits: buttons, camera movements, and the escape button.

### Temperature Discrepancy Between Camera and Buttons
Addressed the issue by decoupling button actions from camera actions within the VPT framework.

### Large Action Space
Reduced an extensive set of button combinations (over 8000) down to the 125 most common actions using an ActEncoderDecoder.

### Downsampling
Opted not to downsample to prevent the loss of crucial actions.

### Excessive Timesteps
Incorporated a GlobalPositionEncoding to handle extended episodes and maintain sequence integrity.

### Reward System
Modified the reward scheme by truncating episodes and assigning an arbitrary reward at the end, particularly for the 'find cave' environment.

### Discount Factor (Gamma)
Maintained a gamma value of 1, as altering it showed negligible effects on performance.

### Hyperparameter Tuning
Configured the model with a modest size and manageable parameters, avoiding overcomplexity:
- 1024 embeddings
- 64 sequence length, tripled for state, action, and rewards_to_go (3 * 64)
- 8 attention heads
- 6 transformer blocks (layers)
- 256 hidden units size
- Inference reward-to-go matched with the end-of-episode reward

### Compatibility Across Implementations
Ensured compatibility with both the Hugging Face Decision Transformer and our custom-developed version.

### Multi Environment Framework
Established a versatile framework capable of handling multiple environments.

## Results

### Proficient Explorer
The agent has mastered navigation, quickly adapting to the environment and overcoming obstacles with ease.

### Task-Specific Learning
The agent's learning is confined to tasks it has experienced; for instance, it has not learned to combat zombies but rather to evade them.

### 'Find Cave' Success Not Solely Attributable to Chance
Empirical evidence suggests that training on the concluding segments of 'find cave' episodes enhances the agent's cave discovery rate beyond mere coincidence.

### Unresolved Environments
Other environments remain unsolved, which is likely attributed to their inherent complexity.

## Conclusions

### The Potency of VPT
The VPT model has proven its strength, with potential applicability in areas like autonomous driving, albeit requiring significant preprocessing resources.

### Sufficient Data for Basic Tasks
The collected videos provide ample data for training basic navigational skills, such as running and jumping.

### Innovations for 'Find Cave' Training
Trimming episode lengths has been effective for training in 'find cave' scenarios, though this method may not universally apply.

### Data Insufficiency for Complex Tasks
Training the Decision Transformer on complex, long-term tasks is challenging due to limited data and the absence of intermediate rewards.

### Hyperparameter Tuning
- Implemented a modestly sized model with the following hyperparameters:
  - **1024 embeddings**: Ensuring sufficient representational capacity.
  - **64 sequence length**: Multiplied by three for state, action, and rewards-to-go, providing a comprehensive context for decision-making.
  - **8 heads**: Allowing the model to attend to different information at different positions.
  - **6 layers (blocks)**: Deep enough to capture complex relationships.
  - **256 hidden size**: Balancing model complexity and computational efficiency.
  - **Reward to go in inference**: Mirrors the end-of-episode reward to guide the prediction.

### Compatible Implementations
- Employed two distinct implementations for versatility and comparison:
  - **Hugging Face Decision Transformer**: Leveraging the established framework for baseline comparison.
  - **Custom Decision Transformer**: Developed in-house for tailored optimization.

### Installation and Running MineRL
- Established a process for installing and executing MineRL to test the Decision Transformer's performance in a diverse gaming environment.

### Multi-Environment Framework
- Developed a framework capable of handling multiple environments, facilitating the testing and comparison of the Decision Transformer's adaptability.

## Results

### Good Explorer
- The agent has adeptly learned to navigate complex environments, showcasing abilities like swift movement, obstacle avoidance, jumping, swimming, and evasion from adversaries.

### Task-Specific Learning
- Observed that the agent acquires proficiency primarily in tasks it has been explicitly exposed to; for instance, it hasn't learned combat strategies, only evasion.

### Beyond Luck in 'Find Cave'
- Post-training analysis indicates that the agent's success in finding caves exceeds what could be attributed to mere chance, suggesting effective learning. Nonetheless, random exploration does play a role due to the nature of the task.

### Unsolved Environments
- Challenges persist in other environments, which remain unresolved—likely due to their intricate complexity.

## Conclusions

### The Power of VPT
- Video PreTraining (VPT) has proven exceptionally potent, with potential applicability in complex fields like autonomous driving.
- Preprocessing with VPT is computationally intensive, demanding substantial GPU resources.

### Sufficient Data for Basic Actions
- The available video data adequately covers fundamental actions such as running, jumping, and obstacle navigation.

### Tactical Episode Trimming
- Strategically cutting episodes to the end phase proved successful for the 'Find Cave' environment but was not universally applicable.

### Insufficiency of Complex Data
- Training a Decision Transformer for long-term, complex actions is challenging with sparse data and a lack of intermediate rewards.

### Sequence Length Considerations
- Extended Decision Transformer sequences do not yield effective results for actions requiring long-term strategic planning without adequate data.

### Streamlined Hyperparameter Optimization
- Hyperparameter tuning for the Decision Transformer is relatively straightforward, with training sessions concluding within an hour and limited parameters requiring adjustments.

## Future Work

### Hierarchical Decision Transformer
- There is a belief that hierarchical structures in Decision Transformers could potentially address the unresolved complexities of other environments, possibly employing a pure or multimodal approach based on the Minedojo framework.
  
![minedojo_hdt](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/0516f842-b7dd-40e6-9352-8580fc8f1be8)

[minedojo.org](https://minedojo.org)

[HIERARCHICAL DECISION TRANSFORMER](https://arxiv.org/pdf/2209.10447.pdf)

# Installation:

You can choose between two ways to install the project: using Docker or directly on an Ubuntu environment with Conda or similar. If you opt for Docker, at the root of the repository, you will find the [dev-container](https://github.com/SwissTonyStark/GameMindsDT/tree/main/.devcontainer/containerMineRL) for installation in Visual Studio, and also in the readme, the steps to follow to build the docker images. Below, you will find how to run the code on Ubuntu.

## In Ubuntu

Install java

```bash
add-apt-repository ppa:openjdk-r/ppa
apt-get update
apt-get install openjdk-8-jdk
```
In a Conda Environment

```bash
# Install the required libraries
pip install -r requirements.txt

# Install the library
pip install -e .
```

## Usage

### Create a settings.yml file:
Create a settings.yml with the following structure, and replace the path_data with the path to the data folder. Example settings.example.yaml:

```yaml
# settings.yaml
path_data: path/to/data
```

### Download the data:

We have created a script to download preprocessed data from our Google bucket for a limited time. Not all environments have been processed. If you need other data or if the bucket is unavailable, you can download and preprocess data from the basalt-benchmark library: (https://github.com/minerllabs/basalt-benchmark).

```bash
python download_data.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0
```

### Go into the folder
```bash
cd to dt_mine_rl
```

### Train the model
```bash
python train.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0

```

### Evaluate the model
```bash
python rollout.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0
```

### The environments
- MineRLBasaltFindCave-v0: FindCaveEnvSpec 
    The only env that seems to work with a standard DT. The others probably need a hirarchical model.
- MineRLBasaltMakeWaterfall-v0: MakeWaterfallEnvSpec,
- MineRLBasaltCreateVillageAnimalPen-v0: PenAnimalsVillageEnvSpec,
- MineRLBasaltBuildVillageHouse-v0: VillageMakeHouseEnvSpec,

## The results

### Cave found - Statistics Results
We have conducted several rounds of rollout and manually checked how many caves are found.

![dt_rollouts](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/a0bbe1d2-6a97-46d9-87b6-757fb8daae83)


### Demo videos

The agent not only learns to find caves. It also learns other basic skills such as escaping from zombies, avoiding obstacles, getting out of traps, exploring, navigating through caves, swimming, etc. Here are some demonstration shortcuts from the many journeys our agent has made.

[hole_in_one.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/3d89bbda-cee0-4db3-8fa4-9f6dfd3207bd)

[ambush.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/5b132767-b715-42b7-89b1-32f39228cf85)

[avoid_the_fall.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/8411bb0e-e0df-401b-a7d8-4958608aafb8)

[driver.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/8c9f05fe-807b-4dba-a3af-9a5df17cfe97)

[found_castle.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/374ddff6-c75b-4abc-8b4d-5f9f002bb966)

[good_view.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/9960baae-1318-4c44-aa75-65a8a4c131af)

[in_house.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/479fa5a6-4b3f-483a-8d35-e873786fb81e)

[more_cave_seeker.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/6c2e37d2-a86e-46ac-9d98-43153ac95fce)

[quick.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/a2a958f2-5b2c-4849-8d98-7e000caf0263)

[seek_cave.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/a105c11e-1248-4ef3-a3fb-4a4b6bcf4e85)

[sometimes_fails.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/904b9867-097e-483d-80f3-b46d26dabdc0)

[zombie_scape.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/edaf9047-f363-4231-a818-80049eea2ed7)

[scape_from_the_archers.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/4d80bc57-04d7-41d7-b961-9fef2e529d91)

[scape_from_trap.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/d0045356-5726-48da-9d01-bf270158342d)

[direct_to_hole.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/c7b8adf5-3555-4816-a4bb-327c438cc819)

[speed_runner.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/4590dfba-0ed8-4622-aa47-87f18f6cced0)

[good_explorer.webm](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/10781cf1-6690-4c95-a697-28ff72f835f4)

## Acknowledgements:
**Mine_rl** MineRL is a rich Python 3 library which provides a OpenAI Gym interface for interacting with the video game Minecraft, accompanied with datasets of human gameplay. 
(https://minerl.readthedocs.io/en/latest/)

**vpt_lib:** OpenAI Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos: (https://github.com/openai/Video-Pre-Training)
- We have used the VPT library to extract the features from the videos and use them as input to the model. 
- We slightly modified the code to fit our needs. Concretly, we have separated the button actions from the camera actions. We have also deactivated inventory actions (for cave search).

**basalt and basalt-benchmark:**
- **Basalt:** NeurIPS 2022: MineRL BASALT Behavioural Cloning Baseline: (https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline)
- **Basalt-benchmark:** (https://github.com/minerllabs/basalt-benchmark)
- We have adapted and reorganized the code from the basalt library to fit our Decision Transformer agent.

**d3rlpy:**
-   **d3rlpy:** A collection of Reinforcement Learning baselines and algorithms for model-based reinforcement learning: We have use his GlobalPositionEncoding. (https://github.com/takuseno/d3rlpy/tree/v2.3.0)

**hugging_face:**
-   **Hugging Face:** We have used the Hugging Face library to use the GPT-2 model and the Decision Transformer model. (https://huggingface.co/docs/transformers/model_doc/decision_transformer)

**other libraries:**
-   We have used other libraries such as numpy, pandas, torch, torchvision, etc.





