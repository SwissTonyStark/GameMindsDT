

# Decision Transformer for MineRL

We have used the [basalt benchmark](https://github.com/minerllabs/basalt-benchmark) framework to test the decision transformers. First, we tested with the Hugging Face Decision Transformer, and then we used our own implementation from scratch. So far, the only environment completed has been find Cave, where we can see that almost 50% of the time the agent ends up finding a cave. Moreover, the agent learns to navigate through the environment with surprising skill. It can get out of complex situations and avoid obstacles. We believe that for the other environments, it is necessary to create a Hierarchical Decision Transformer.

## Decision Transformer only trained with videos
This model learns to make decisions based on the features extracted from the human video frames. The features are extracted using the VPT library. This means that it learns to navigate through Minecraft and to find caves having been trained solely and exclusively with the viewing of videos (thanks to the embeddings and the extraction of actions from the VPT library). Additionally, for the model to understand that we wanted it to find caves, the videos have been trimmed so that only a few frames from the end have been used for training. 

The only techniques used were to disable the inventory button and to cut the videos. An arbitrary reward has been added at the end of the episodes. No additional information has been added.

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

### The results


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


## Results

Rollout 1:
7/12

Rollout 2:
5/12

Rollout 3:
6/12

Rollout 4:

