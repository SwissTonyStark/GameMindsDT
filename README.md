# GameMindsDT - Project Overview

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/232c06e1-7cb4-4d0b-b2ae-11d24c79a2ef" alt="GameMindstDT Atari & Minecraft" width="896" height="512">
</p>

## Introduction and Motivation 🌟
Welcome to **GameMindsDT**, where we combine the potent synergy of **Decision Transformers** and **Reinforcement Learning** to conquer a myriad of gaming challenges. Our project is the epic saga of constructing a **custom Transformer model** from scratch, fine-tuning reinforcement learning models, and deploying them to triumph in an extensive variety of virtual environments. From the nostalgia of classic arcade arenas to the intricate strategies of sophisticated simulations, **GameMindsDT** stands as a vanguard in **AI-driven gameplay exploration**.

### Reinforcement Learning: A Primer 📚
**Reinforcement learning (RL)** is a specialized branch of machine learning that empowers an agent to learn by perceiving and interpreting its environment. The agent operates within this environment, striving to maximize a reward function tied to its actions.
In conventional scenarios, RL algorithms operate online, learning directly through environmental interaction. This iterative learning process involves the agent receiving environmental observations, taking actions, and earning rewards, which leads to the discovery of the next state.

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/ae47e1ce-a98a-4348-a918-3505bbba5206" alt="RL Image">
</p>

However, the challenge arises when a simulation environment is unavailable. Constructing such an environment can be complex and resource-intensive.

### The Offline RL Paradigm 🔁
Conversely, in **offline RL**, an agent leverages pre-collected data, bypassing the need for direct environmental interaction. This data, typically stored as sequences, may originate from humans or other agents. The pivotal concern here is data quality; subpar or insufficient data can severely impact the optimization of the agent's policy.

> # Table of Contents 📖
> - [Objectives and Hypothesis](#objectives-and-hypothesis)
> - [Project Management](#project-management)
>   - [Team Composition and Work Distribution](#team-composition-and-work-distribution)
>   - [Gantt Chart and Milestones](#gantt-chart-and-milestones)
> - [State of the Art: Decision Transformer](#state-of-the-art-decision-transformer)
> - [Environments](#environments)
> - [Installation and Experiments](#installation-and-experiments)
>   - [MineRL: Exploration and Objectives](#minerl-exploration-and-objectives)
> - [Docker Integration](#docker-integration)
> - [Resources](#resources)
> - [Conclusions](#conclusions)
> - [References and Acknowledgements](#references-and-acknowledgements)
> - [Licence](#license)
> - [Glossary](#glossary)

# Objectives and Hypothesis 🎯
Our goal was to explore the realms of AI in gaming beyond traditional approaches, hypothesizing that Decision Transformers can provide a more nuanced understanding and execution of game strategies. We aimed to unlock the untapped potential of these transformers across a broad spectrum of games and tasks, pushing the limits of AI capabilities in virtual environments.

## **Exploration of the Decision Transformer** 🚀

### **1. Decision Transformer**

#### **1.1. Analysis and Understanding of the Model**
- Conduct an in-depth analysis to understand how the Decision Transformer works.

#### **1.2. Implementation of the Model**
- Follow the official paper to implement the model from scratch.

### **2. Variants of the Decision Transformer**

#### **2.1. Analysis and Implementation of Variants**
- Explore and analyze variants of the original model, such as the **Hierarchical Decision Transformer** or the **Constrained Decision Transformer**.
- Implement these variants and compare them with the base model.

### **3. Performance in Classic Environments**

- Assess the performance of the Decision Transformer in classic RL environments, like **Atari** or **Mujoco**.
- Compare its performance against classic RL algorithms.

### **4. Performance in Complex Environments**

- Check the model's performance in more complex environments, using **Minecraft** as a simulation environment.

# Project Management 🛠️
Regarding the planning, it has been divided into two different paths, forming two teams to tackle the proposed objectives:
- The first team is responsible for evaluating the performance of various RL algorithms alongside the DT, testing various existing environments for DT implementation.
The paths converge in the final stages of the project to merge progress and share insights, with the ultimate goal of testing the DT implementation in the Minecraft environment, specifically using MineRL.
- The second team focuses on analyzing the original paper and implementing a Decision Transformer (DT) from scratch, including all tasks related to model definition and training.

## Team Composition and Work Distribution 🧑‍🤝‍🧑
**Team A:** Research and testing current available environments and Docker container development for training the Decision Transformer in MineRL environment
- [Pol Fernández Blánquez](https://www.linkedin.com/in/polfernandezblanquez/)
- Edgar Planell

**Team B:** Research and creating a Decision Transformer from scratch, for testing and comparing it with different experiments
- Shuang Long Ji Qiu
- Omar Aguilera Vera
- Alex Barrachina

## Gantt Chart and Milestones 📅
Our project timeline and key milestones were meticulously tracked using a Gantt chart, illustrating our structured approach to reaching our objectives.

<img width="1087" alt="Gannt" src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/d640f614-b9a4-4b8c-be01-b9ca77b8c669">


# State of the Art: Decision Transformer 🌐
The leitmotif of this project is centered on **Reinforcement Learning**. However, focusing on it in an offline manner and, as previously mentioned, working with data sequences and not directly interacting with an environment. This is why this project explores the architecture of the **Decision Transformer** presented in this paper, where it reduces the RL problem to a conditional sequence modeling problem. As its name suggests, it is based on **Transformers**, models _par excellence_ for solving sequence problems.

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/10a57bff-849c-4d44-985b-4f72c83c8d03" alt="Decision Transformer" width="545" height="335">
</p>

Unlike basic RL algorithms that are based on estimating a function or optimizing policies, DTs directly model the relationship between **cumulative reward** (return-to-go), states, and previous actions, thus predicting the future action to achieve the desired reward.
As already mentioned in the paper, the generation of the next action is based on **future desired returns**, rather than past rewards. That's why, instead of using the rewards space, so-called **return-to-go** values are fed with the states and actions.

_Return-to-go_ is nothing more than the total amount of reward that is expected to be collected from a point in time until the end of the episode or task. If you are halfway through your journey, the return to go is the sum of all future rewards that you expect to earn from that midpoint to the destination. Then, the main goal of the agent will be to **maximize this value**, performing actions such that at the end of the episode it has achieved a future reward.

The architecture is simple; as input, we feed the DT with the **return-to-go**, state, and action. Each of these three spaces is tokenized, having a total of three times the length of each space. So, if we feed the last _K_ tokens to the DT, we will have a total of **3*K tokens**. To obtain the token embeddings, we project a linear layer for each modality, but in the case we are working with visual inputs, **convolutionals** are used instead.

In addition, the Decision Transformer introduces a unique approach to handle sequence order. An **embedding of each timestep** is generated and added to each token. Since each timestep includes states, actions, and return-to-go, the traditional positional encoding can’t be applied because it won’t guarantee the actual order. Once the input is tokenized, it is passed as input to a **decoder-only transformer** (GPT).

In the following block, we can see the pseudocode from the Decision Trasnformer paper. It basically describes the implementation about the DT, and a possible train and eval loop:

```python
# Algorithm 1 Decision Transformer Pseudocode (for continuous actions)

# R, s, a, t: returns-to-go, states, actions, or timesteps
# transformer: transformer with causal masking (GPT)
# embed_s, embed_a, embed_R: linear embedding layers
# embed_t: learned episode positional embedding
# pred_a: linear action prediction layer

# main model
def DecisionTransformer(R, s, a, t):
    # compute embeddings for tokens
    pos_embedding = embed_t(t)  # per-timestep (note: not per-token)
    s_embedding = embed_s(s) + pos_embedding
    a_embedding = embed_a(a) + pos_embedding
    R_embedding = embed_R(R) + pos_embedding

    # interleave tokens as (R_1, s_1, a_1, ..., R_K, s_K)
    input_embeds = stack(R_embedding, s_embedding, a_embedding)

    # use transformer to get hidden states
    hidden_states = transformer(input_embeds=input_embeds)

    # select hidden states for action prediction tokens
    a_hidden = unstack(hidden_states).actions

    # predict action
    return pred_a(a_hidden)

# training loop
for (R, s, a, t) in dataloader:  # dims: (batch_size, K, dim)
    a_preds = DecisionTransformer(R, s, a, t)
    loss = mean((a_preds - a)**2)  # L2 loss for continuous actions
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# evaluation loop
target_return = 1  # for instance, expert-level return
R, s, a, t, done = [target_return], [env.reset()], [], [1], False
while not done:  # autoregressive generation/sampling
    # sample next action
    action = DecisionTransformer(R, s, a, t)[-1]  # for cts actions
    new_s, r, done, _ = env.step(action)

    # append new tokens to sequence
    R = R + [R[-1] - r]  # decrement returns-to-go with reward
    s, a, t = s + [new_s], a + [action], t + [len(R)]

    # only keep context length of K
    R, s, a, t = R[-K:], s[-K:], a[-K:], t[-K:]
```
# Environments 🎮
We explored a variety of algorithms and environments, ranging from OpenAI Gym to more complex simulations, each presenting unique challenges and learning opportunities.

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/4e11aaf8-9ae9-4721-841e-db724e772de4" alt="Environments GameMinds DT" width="700" height="700">
</p>

# Installation and Experiments 🔧
For installation instructions and detailed experiment walkthroughs, refer to the specific README files linked below. Here's a quick start:

```bash
git clone https://github.com/SwissTonyStark/GameMindsDT.git
cd GameMindsDT
pip install -r requirements.txt
```
## Preliminary Experiments 🚀
Before diving into our custom Decision Transformer, we first aimed to validate the algorithm's effectiveness against classical RL algorithms. These initial tests utilized an existing Decision Transformer implementation from the d3rlpy library. For a detailed overview, refer to our [Preliminary Experiments README](experiments/README.md).

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/d05b3385-73c9-4abb-a1d0-c03e06b9646f" alt="key_to_door">
</p>

*Here, the agent is tasked with finding a key to unlock the door, progressing to the next level.*

## Atari: Exploration and Objectives 🕹️
<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/01c5365c-5b41-4ac1-8ca2-96e37aa74294" alt="Atari logo" width="200" height="103">
</p>
We trained models on Atari games using a Decision Transformer to benchmark our approach against this visually rich and diverse environment, specially challenging due to the difficulty of credit assignment arising from the delay between actions and resulting rewards.
For a detailed overview, refer to our <a href="https://github.com/SwissTonyStark/GameMindsDT/blob/main/DT-atari/README.md" target="_blank">DT-Atari README</a>


<table style="padding:10px">
  <tr>
    <td><img src="/assets/seaquest_3620.gif"  alt="1" width = 350px height = 496px ></td>
     <td><img src="/assets/spaceinvaders_1350.gif"  alt="1" width = 350px height = 496px ></td>
  </tr>
</table>

### PyBullet: Exploration and Objectives
<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/bullet.png" alt="Minecraft" width="512" height="150">
</p>
In this experiment, we are utilizing our Decision Transformer in PyBullet environments through the d4rl-pybullet library from this repository. Initially, the plan was to employ the MuJoCo environment from Gymnasium, but due to issues installing the library stemming from deprecated dependencies and licensing problems, we sought an alternative environment.

The d4rl-pybullet library features four replicable environments: Hopper, HalfCheetah, Ant, and Walker2D. The primary objective is to assess how our decision transformer performs in these types of environments. For an in-depth examination, please visit the [DT-PyBullet README](d4rl_pybullet_dt/README.md)

## MineRL: Exploration and Objectives ⛏️
<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/d75127fd-e5a2-4f32-b0a4-2b6ec87778a8" alt="Minecraft" width="512" height="150">
</p>

Pushing the boundaries further, we applied the Decision Transformer to Minecraft via the MineRL environment. This experiment aimed to explore the model's potential in complex, real-world tasks based on human video demonstrations. Dive deeper into our findings in the [MineRL README](dt-mine-rl-project/README.md).

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/d6d4af15-0e29-4013-a6bb-d38f74e09921" alt="Hooe in one" width="" height="">
</p>

*Minecraft's complexity presents a significant challenge, testing the Decision Transformer's learning efficiency and adaptability.*

## Docker Integration 🐳
## Introduction to Docker

Docker played a crucial role in ensuring a consistent development environment across our team. It's an *open-source containerization platform* that enables developers to **package applications into containers—standardized executable components** combining application source code with the operating system (OS) libraries and dependencies required to run that code in any environment.
<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/acba1b48-351c-4977-9283-5dacf8115b14" alt="Docker logo" width="640" height="150">
</p>

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
Then you will have installed the base image to run our project in a container! But keep reading, you will need to add some extensions, new images which contain more dependencies.

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
Once you have built the image you need, you can run a container based on that image. Make sure to replace `nvidia-pytorch:base`, `minerl-dt`, or `pybullet-dt` with the name of the image you have built. You will find your images in Docker Desktop, it's recommendable to check if you have them all, but you won't need to run them manually. Visual Studio Code does it for you, it runs an instance of them.

#### Setting Up Development Environment in VSCode
After building the images, open the project in Visual Studio Code. Use the "Reopen in Container" feature from the top menu. Select the desired container from the available options.

The Docker containers are defined in the `./devcontainer` directory, where you can find their configurations.

That's it! You are now ready to work with Docker and use the built images for your projects with MineRL and PyBullet.

## Resources
Overview of our repository structure and data flow diagrams to navigate through our project's architecture efficiently.

>![image](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/42be917e-72f8-49d8-97b7-57624d1b7a2e)

## Conclusions 🌐

Reflecting on our GameMindsDT project, moving from the start to finish has been both a big challenge and a great learning experience. Using Decision Transformers in classic games like Atari was done with hope and careful planning as we moved forward in AI research. We noticed a lack of detailed guides and real examples, which showed we were doing something new and exciting.

The project showed us how flexible Decision Transformers can be, especially in situations where we only have data to learn from and no real game interaction. Depending on a lot of well-organized data was a big hurdle, highlighting the gap between having enough simple data and not having enough for more complex tasks.

Adding our improvements to the Decision Transformer model led to good results, showing the importance of constant creativity and making changes. Despite facing expected difficulties with Docker and dealing with many dependencies, Docker turned out to be very helpful. It made working together easier, made our workflows smoother, and made sure everyone had the same development environment, which was key for our teamwork.

In the end, the GameMindsDT project didn't just push the limits of using AI in gaming but also gave us valuable insights into using new technologies. It strengthened our belief in the power of AI and machine learning. Despite the challenges, it made us more determined to keep going in this direction, sparking new ideas and extending the use of AI beyond gaming to more complex areas.

**Key Takeaways:**
- **Success with Decision Transformers**: Our custom changes have led to significant success, showing what's possible with specialized AI models.
- **The Benefit of Docker**: Despite some setup challenges, Docker was extremely useful, making it easier for our team to work together. Its ability to manage work from multiple developers was clearly a big plus.


## License
This project is licensed under the [MIT License](LICENSE).

## Glossary

**Decision Transformers (DT)**: A novel approach to reinforcement learning that frames the RL problem as a conditional sequence modeling task, leveraging Transformer architectures to predict future actions based on past states, actions, and desired future rewards. [More info](https://arxiv.org/abs/2106.01345)

**Reinforcement Learning (RL)**: A branch of machine learning where an agent learns to make decisions by performing actions in an environment to maximize some notion of cumulative reward. [More info](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)

**Offline Reinforcement Learning**: A paradigm of reinforcement learning where the agent learns from a fixed dataset of previously collected experiences without further interaction with the environment. [More info](https://arxiv.org/abs/2005.01643)

**Return-to-Go**: The total expected reward that an agent can accumulate from a certain point in time until the end of the episode. This concept is crucial in many RL algorithms, including Decision Transformers. [More info](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Atari**: A classic set of environments provided by the OpenAI Gym toolkit, based on Atari 2600 video games, commonly used for evaluating reinforcement learning algorithms. [More info](https://gym.openai.com/envs/#atari)

**Mujoco**: A physics engine used for simulating complex robot dynamics and kinematics, often utilized in reinforcement learning experiments for tasks requiring physical interactions. [More info](https://mujoco.org/)

**Minecraft**: A popular sandbox video game that serves as a complex, open-world environment for testing advanced reinforcement learning models, such as those capable of navigating and manipulating their environment. [More info](https://www.microsoft.com/en-us/research/project/project-malmo/)

**Docker**: An open-source platform for developing, shipping, and running applications in containers, which allows for packaging an application and its dependencies into a standardized unit for software development. [More info](https://www.docker.com/)

**Gantt Chart**: A type of bar chart that illustrates a project schedule, representing the start and finish dates of the various components and elements of a project. [More info](https://www.projectmanager.com/gantt-chart)

**CUDA**: A parallel computing platform and application programming interface (API) model created by Nvidia, allowing software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing (an approach known as GPGPU, General-Purpose computing on Graphics Processing Units). [More info](https://developer.nvidia.com/cuda-zone)

**PyTorch**: An open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). [More info](https://pytorch.org/)

**MineRL**: A research project aimed at solving open problems in artificial intelligence and machine learning using the popular game Minecraft as a platform for developing intelligent agents. [More info](https://minerl.io/)

**Hierarchical Decision Transformer**: A variant of the Decision Transformer that introduces a hierarchical structure into the model to handle more complex decision-making scenarios with longer sequences or multiple objectives. While specific papers on "Hierarchical Decision Transformers" might not be available, the concept builds on principles found in both decision transformers and hierarchical reinforcement learning. [More info on HRL](https://arxiv.org/abs/1604.06057)

**Constrained Decision Transformer**: Another variant of the Decision Transformer designed to operate under specific constraints, optimizing the policy within given boundaries to ensure safe or compliant behavior in sensitive environments. Like the Hierarchical Decision Transformer, specific references might be conceptual, reflecting advancements in constrained RL and decision transformers. [More info on Constrained RL](https://arxiv.org/abs/2005.00513)

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

# Future work
### Evolution of the Decision Transformer in our project
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Hierarchical Decision Transformer](https://arxiv.org/abs/2209.10447)
- [Elastic Decision Transformer](https://kristery.github.io/edt/)
- [Critic-Guided Decision Transformer](https://arxiv.org/abs/2312.13716)
- [Online Decision Transformer](https://arxiv.org/pdf/2202.05607.pdf)
- [Constrained Decision Transformer a](https://www.offline-saferl.org/)
- [Constrained Decision Transformer b](https://arxiv.org/abs/2302.07351)
> **Published in: 2023-2024**
> ![GameMindsDT Cover](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/c36ded66-414c-48c1-9700-9cf727359b41)
