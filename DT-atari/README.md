# Decision Transformer for Atari

## Motivation
We decided to train Atari games using a decision transformer for some reasons:
- <b>Complexity</b>: Atari games offer a visually rich and diverse environment, in some cases more complex than other more simpler environments often used in RL. Some Atari games are specially challenging due to the difficulty of credit assignment arising from the delay between actions and resulting rewards
- <b>Benchmarking</b>: Atari games serve as a well-established benchmark for RL algorithms, that allows us to compare the performance of owr model with other RL techniques.

## Implementation
We implemented from scratch another Decision Transformer to try to solve Atari games. This DT is very similar to the previous one, except that:
- convolutional encoder: In a transition, the state is an image of the game screen. In fact, a stack of images of the last 4 transitions (to catch information like velocity objects). The state is fed into a convolutional encoder instead of a linear layer to obtain token embeddings. The fact that the states are images and take up much more space complicates the data management.
- cross-entropy loss function: An Atari game typically have a discrete set of actions (e.g., jump, move left, shoot). So, we use the cross-entropy loss function instead of the mean-squared error. 

<img src="https://cosmoquester.github.io/assets/post_files/2021-06-29-decision-transformer/Untitled.png"  alt="1" width = 700px height = 496px >


We use a different version of gym and a dedicated version of D4rl, to have access to the Atari datasets

The original paper uses Tanh (hyperbolic tangents) instead of LayerNorm  after the embeddings, but we obtained the same or worst results using Tanhs.
We adapted the position encoding with an alternative version that improves the performance thanks to the d3lply library.

We have used hyperparameter values similar to those of the original paper:
- embedding dimension 128
- layers (blocks) 6
- attention heads 8
- context length 30
- Return-to-go conditioning 
  - Qbert 2500
  - Seaquest 1450
  - Space Invaders 20000
  - Breakout 90

## Results

We trained the Breakout, Qbert, Space_invaders and Seaquest.
In the evaluation, As the reward may vary between episodes, after every epoch we calcule the mean of the total rewards of 100 episodes.
We trained the model with a dataset of 1.000.000 trajectories and we observed that the model improves the scores til aproximately 4/5 epochs.

<img src="/assets/metrics_atari.png"  alt="1" width = 350px height = 496px >

Atari games perform better when trained with expert data. We trained Breakout with mixed data and the performance is not as good as the other games. We trained Qbert with mixed and expert data and the scores are 6 times better with expert data.

At test time the DT is handed the first return-to-go token, indicating the desirable return to reach in the task. Til aprox. 5 times the maximum return-to-go in the dataset, he results show a correlation between desired target return and true observed return. We used higher initial return-to-go with a similar result as 5x the maximum in the dataset.

Some examples of the performance after training:

<table style="padding:10px">
  <tr>
    <td><img src="/assets/seaquest_3620.gif"  alt="1" width = 350px height = 496px ></td>
    <td><img src="/assets/qbert_19000.gif"  alt="1" width = 350px height = 496px ></td>
  </tr>
  <tr>
     <td><img src="/assets/spaceinvaders_1350.gif"  alt="1" width = 350px height = 496px ></td>
      <td><img src="/assets/breakout_32.gif"  alt="1" width = 350px height = 496px ></td>
  </tr>
</table>

Except in the case of the Breakout (trained only with mixed data) we observe that our DT obtains similar, or in some cases even higher scores than those obtained in our previous preliminar comparative using the DT of D3rl. We ran 100 episodes of each game for each algorithm and we averaged the results.

![atari_results](/assets/atari_comparative.png)

## Future work

- Cook the data better: Include to our dataset information of the current number of lives over the course of an episode. It's relevant information. We could try to train the model not to lose lives, as well as to accumulate the maximum reward.
- Crop the images (states) in patches to retain positional information, as in ViTs.
- We changed the positional encoding with success. We could try new position encoding strategies like AliBi or RoPE.

Some of these proposals are addressed in a variant of the DT called  Multi-game DT, that we started to implement (and which we explain below)

## Installation:

Installation options:

1/Just run our colab:
[DT-atari](https://colab.research.google.com/drive/1KhUZoDw-3lbm41GLiuoboDp6pn4dfAoe?usp=sharing)

2/Local installation:
Dependencies can be installed with the following commands:

```
conda create --name dt python=3.10.12
conda activate dt
pip install git+https://github.com/takuseno/d4rl-atari
pip install 'gym [atari,accept-rom-license]==0.25.2'
pip install -r requirements.txt
```

Example usage

Scripts to reproduce our Decision Transformer results can be found in `train_.sh`.

By default:
```
python ./code/main.py
```


## Multi-game Decision Transformer

As an extra exercise, we decided to explore the multi-game decision transformer, a variant of the DT that tries to produce generalist reinforcement learning agents, trained to play simultaneously on up to 46 Atari games with a single transformer-based model (with a single set of weights). As we are interested in the transfer ability of pretained agents to new games, we tried to fine-tuning a novel game (that is not in the original dataset) using a checkpoint of a pre-trained multi-game DT. 

We train the model to predict the next return, action, and reward discrete token in a sequence via standard cross-entropy loss. While we may not directly use all of the predicted quantities, the task of predicting them encourages structure and representation learning of the environments.
As the DT has to serve a variety of environments, we standardized the possible actions, rewards and returns-to-go in a uniform quantization. We divide each observation image into a collection of M patches in a similar way that in ViTs. Each patch is additively combined with a trainable position encoding and linearly projected into the input token embedding space. 

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhA0HI4agTrV67FzYHiM02BPlK2Ez61uiktVeSNuLii_C7Tuf1V5hJXmblx5HIFmX7PbT9jhPhNvafM7GZIILSPzY690Sb6O75231kBdXcgBHT1lTFEzVpZaY_ngqtAZs-VspXscd5x30hH8k62sr8GLZeLVX6qGnFeSjcqAIdS85jNjZUIQzoeUn2dLw/s1280/image5.png">

On inference-time, instead of manually fixing the initial rewards-to-go of the entire episode, we use kind a variation of return-conditioned policies that automatically generates expert-level returns at every timestep, using a binary classifier that identifies whether or not the behavior is expert-level before taking an action at time t. 

Currently we have managed to adapt the original code so that the model starts training a new game (fine-tuning a pretrained model), but we are still missing the evaluation function.
You can see our progress in this colab:
[Multi-game-DT fine-tuning](https://colab.research.google.com/drive/1SbHa5kluw8BnfsCjitUErp0XlpWO5Lv9#scrollTo=3C8ay4GGDWI9)

## Acknowledgements:

Datasets of Atari games

https://github.com/takuseno/d4rl-atari

Alternative global position encoder

https://github.com/takuseno/d3rlpy

Decision transformer

We inspired our implementation on the following repositories:
- https://github.com/kzl/decision-transformer
- https://github.com/nikhilbarhate99/min-decision-transformer
other important references:
https://arxiv.org/pdf/2106.01345.pdf
https://huggingface.co/docs/transformers/main/en/model_doc/decision_transformer
https://huggingface.co/edbeeching/decision_transformer_atari

Multi-game DT

https://openreview.net/references/pdf?id=LCr7cYytgP
https://openreview.net/references/attachment?id=plw8BUq0F_&name=supplementary_material
https://github.com/etaoxing/multigame-dt
https://github.com/google-research/google-research/tree/master/multi_game_dt
https://github.com/luciferkonn/MDT


