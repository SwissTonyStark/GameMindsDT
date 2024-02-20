
Installation:


```bash

# Install the required libraries
pip install -e requirements.txt

# Install the library
pip install -e .
```

## Usage

```bash
Create a settings.yml with the following structure:
```yaml
# settings.yml
path_data: path/to/data
```

### Download the data: TODO
```bash
python download_data.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0
```

## Go into the folder
```bash
cd to dt_mine_rl
```

### Train the model
```bash
```bash
python train.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0

```

### Evaluate the model
```bash
python rollot.py --env=[env] # Where env is the name of the environment, Example: MineRLBasaltFindCave-v0
```

## The environments
    - MineRLBasaltFindCave-v0: FindCaveEnvSpec,
    - MineRLBasaltMakeWaterfall-v0: MakeWaterfallEnvSpec,
    - MineRLBasaltCreateVillageAnimalPen-v0: PenAnimalsVillageEnvSpec,
    - MineRLBasaltBuildVillageHouse-v0: VillageMakeHouseEnvSpec,

## Aknouledgements:
    vpt_lib: OpenAI Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos: https://github.com/openai/Video-Pre-Training
        - We have used the VPT library to extract the features from the videos and use them as input to the model. 
        - We slightly modified the code to fit our needs. Concretly, we have separated the button actions from the camera actions.

    basalt:
        - Basalt: NeurIPS 2022: MineRL BASALT Behavioural Cloning Baseline: https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline
        - We have adapted and reorganized the code from the basalt library to fit our Decision Transformer agent.

    d3rlpy:
        -   d3rlpy: A collection of Reinforcement Learning baselines and algorithms for model-based reinforcement learning: We have use his GlobalPositionEncoding. https://github.com/takuseno/d3rlpy/tree/v2.3.0

    hugging_face:
        -   Hugging Face: We have used the Hugging Face library to use the GPT-2 model and the Decision Transformer model. https://huggingface.co/docs/transformers/model_doc/decision_transformer
    
    other libraries:
        -   We have used other libraries such as numpy, pandas, torch, torchvision, etc.