

# Decision Transformer for MineRL

We have used the [basalt benchmark](https://github.com/minerllabs/basalt-benchmark) framework to test the decision transformers. First, we tested with the Hugging Face Decision Transformer, and then we used our own implementation from scratch. So far, the only environment completed has been find Cave, where we can see that almost 50% of the time the agent ends up finding a cave. Moreover, the agent learns to navigate through the environment with surprising skill. It can get out of complex situations and avoid obstacles. We believe that for the other environments, it is necessary to create a Hierarchical Decision Transformer.

## Decision Transformer only trained with videos
This model learns to make decisions based on the features extracted from the human video frames. The features are extracted using the VPT library. This means that it learns to navigate through Minecraft and to find caves having been trained solely and exclusively with the viewing of videos (thanks to the embeddings and the extraction of actions from the VPT library). Additionally, for the model to understand that we wanted it to find caves, the videos have been trimmed so that only a few frames from the end have been used for training. 

The only techniques used were to disable the inventory button and to cut the videos. An arbitrary reward has been added at the end of the episodes. No additional information has been added.

## How it works
### VPT

![vpt_schema](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/dabd6445-2d82-4a65-94df-8969acd390d2)

### Training
![training_dt](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/4983723a-06de-4a34-a3d5-829b54067e90)

### Playing
![playing_dt](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/ad7ddafc-f320-4c27-a664-e339c0b7d68c)


*Credits for some images*: https://cdn.openai.com/vpt/Paper.pdf, https://arxiv.org/pdf/2106.01345.pdf, https://www.minecraft.net/

## Main challenges and Decissions   

* 3 action logits (buttons, camera, esc button)
Implement a Multicategorical distribution class
* Problems with temperature of camera vs buttons
Decouple the button action from the camera action in VPT.
* Big action space buttons combinations (> 8000)
ActEncoderDecoder reducer to most commons 125
* Downsampling
Disabled, we may miss important actions
* Timesteps too long
Added a GlobalPositionEncoding (max_episode_length, sequence_length)
* Not Rewards
Cut episodes and give and arbitrary reward at the end (for find cave env)
* Gamma = 1
Changing the value doesn't seem to have an impact
* Hiperparameter tunning
Not a big model
- 1024 embeddings
- 64 sequence length ( * 3 (state,action,rewards_to_go))
- 8 heads
- 6 layers (blocks)
- 256 hidden size
- Reward to go in inference
* The same as the end-of-episode reward
Two compatible implementations
- Hugging Face DT
- Our own DT
* Install and run mine RL
* Multi Environment framework

## Results

* Good Explorer
The agent learns to navigate the environment without difficulties. Learns to go fast, avoid trees, jump over steps, swim, climb hills, get out of puddles, run away from enemies, …
* Learns only seen tasks
The agent learns only tasks it has seen, for example, it doesn't know how to fight zombies, only how to escape from them.
* Not only luck in Find Cave
It's easy to see that after training it with the end of episodes in "find cave", the agent finds more caves than by pure chance. However, there's always a luck component due to blind exploration.
* The other environments have not been resolved.
Despite our attempts, we have been unable to solve the other environments. We suspect it's due to their complexity.

## Conclusions

* VPT is Powerful
VPT is a very powerful model, and it's likely exportable to projects such as autonomous driving or similar.
Preprocessing VPT is expensive
It requires a day's worth of GPU work per environment.
* Enough data for simple tasks.
With the videos, we have big data for simple actions (run, jump, avoid obstacles, climb, etc).
* Find caves needs a trick
Cutting episodes has been effective for this case but doesn't work in other cases.
* Not enough data for complex tasks.
It's not easy to train a decision tree (DT) on long term actions without intermediate rewards and low data availability. There are big data for simple actions, but low data for complex actions such building houses.
* Long DT sequences not work.
Long sequences of DT don't work for long term actions with low data and not intermediate rewards.
* Training and finding DT hyperparameters is relatively simple
Less than an hour per training round, and few parameters to adjust with not many differences.

Future Work

* Hierarchical Decisión transformer
We believe that the other environments could probably be solved with a hierarchical DT, which could be pure or multimodal based on Minedojo




# Installation:

## In Ubuntu Install java

```bash
add-apt-repository ppa:openjdk-r/ppa
apt-get update
apt-get install openjdk-8-jdk
```

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

**Rollout 1:**
7 caves of 12 tries

**Rollout 2:**
5 caves of 12 tries

**Rollout 3:**
6 caves of 12 tries

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





