# GameMindsDT - Project Overview
## Introduction and Motivation
Welcome to **GameMindsDT**, where we combine the potent synergy of **Decision Transformers** and **Reinforcement Learning** to conquer a myriad of gaming challenges. Our project is the epic saga of constructing a **custom Transformer model** from scratch, fine-tuning reinforcement learning models, and deploying them to triumph in an extensive variety of virtual environments. From the nostalgia of classic arcade arenas to the intricate strategies of sophisticated simulations, **GameMindsDT** stands as a vanguard in **AI-driven gameplay exploration**.

### Reinforcement Learning: A Primer
**Reinforcement learning (RL)** is a specialized branch of machine learning that empowers an agent to learn by perceiving and interpreting its environment. The agent operates within this environment, striving to maximize a reward function tied to its actions.
In conventional scenarios, RL algorithms operate online, learning directly through environmental interaction. This iterative learning process involves the agent receiving environmental observations, taking actions, and earning rewards, which leads to the discovery of the next state.
![image](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/ae47e1ce-a98a-4348-a918-3505bbba5206)
However, the challenge arises when a simulation environment is unavailable. Constructing such an environment can be complex and resource-intensive.


### The Offline RL Paradigm
Conversely, in **offline RL**, an agent leverages pre-collected data, bypassing the need for direct environmental interaction. This data, typically stored as sequences, may originate from humans or other agents. The pivotal concern here is data quality; subpar or insufficient data can severely impact the optimization of the agent's policy.

Let's continue with what we will see next.

# Table of Contents
- [Objectives and Hypothesis](#objectives-and-hypothesis)
- [Project Management](#project-management)
  - [Team Composition and Work Distribution](#team-composition-and-work-distribution)
  - [Gantt Chart and Milestones](#gantt-chart-and-milestones)
- [State of the Art: Decision Transformer](#state-of-the-art-decision-transformer)
- [Algorithms and Environments](#algorithms-and-environments)
- [Installation and Experiments](#installation-and-experiments)
  - [MineRL: Exploration and Objectives](#minerl-exploration-and-objectives)
- [Docker Integration](#docker-integration)
- [Resources](#resources)
- [Conclusions](#conclusions)
- [References and Acknowledgements](#references-and-acknowledgements)
- [Licence](#license)
- [Glossary](#glossary)

# Objectives and Hypothesis
Our goal was to explore the realms of AI in gaming beyond traditional approaches, hypothesizing that Decision Transformers can provide a more nuanced understanding and execution of game strategies. We aimed to unlock the untapped potential of these transformers across a broad spectrum of games and tasks, pushing the limits of AI capabilities in virtual environments. The main objectives are:

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

# Project Management
Regarding to the planning, it has been divided into two different paths, forming two teams to tackle the proposed objectives:
The first team aims to analyze the original paper and implement a Decision Transformer (DT) from scratch. This includes carrying out all the tasks to train the model: model definition and training.
The second team is responsible for checking the performance of different RL algorithms along with the DT, testing the various existing environments that can be used for the DT implementation.
Finally, in the last steps of the project, these paths converge at the same point to merge the progress and share it. The last objective is to test the DT implementation with the Minecraft environment, specifically using MineRL.

### Team Composition and Work Distribution
#### Supervisor
- **Txus Bach**
#### Team A
- **[Pol Fernández Blánquez](https://www.linkedin.com/in/polfernandezblanquez/)**
- **Edgar Planell**
#### Team B
- **Shuang Long Ji Qiu**
- **Omar Aguilera Vera**
- **Alex Barrachina**

### Gantt Chart and Milestones
Our project timeline and key milestones were tracked using a Gantt chart, illustrating our structured approach to achieving our objectives.

<!--DIAGRAMA GANNT -->
![image](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/a9e6e770-8ec6-4e02-a5be-fab2d814f058)

Over the course of this project, we systematically approached the exploration and development of the Decision Transformer (DT). Our journey began with an in-depth analysis of existing DT environments and models, where Team A achieved complete progress, setting a robust foundation for the project. Subsequently, Team B took the reins to develop and optimize our DT from the ground up, ensuring it was tailored to our specific needs and achieving full progress in all related tasks. **AFEGIM asteriscs amb problemes o les tasques amb menys %??**

# State of the Art: Decision Transformer
The leitmotif of this project is centered on **Reinforcement Learning**. However, focusing on it in an offline manner and, as previously mentioned, working with data sequences and not directly interacting with an environment. This is why this project explores the architecture of the **Decision Transformer** presented in this paper, where it reduces the RL problem to a conditional sequence modeling problem. As its name suggests, it is based on **Transformers**, models _par excellence_ for solving sequence problems.
![image](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/10a57bff-849c-4d44-985b-4f72c83c8d03)
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

# Algorithms and Environments
We ventured through various algorithms and environments, from the traditional OpenAI Gym settings to complex strategic simulations, each offering unique challenges and learning opportunities.

<!-- DIAGRAMA AMB ELS LLISTATS D'ALGORISMES UTILITZATS I ENVIRONMENTS -->
<!-- DRAW.IO -->
![Environments GameMinds DT](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/4e11aaf8-9ae9-4721-841e-db724e772de4)

 *These are the main environments we have tested and experimented with*


 
# Installation and Experiments
For installation instructions and detailed experiment walkthroughs, refer to the specific README files linked below. Here's a quick start:

```bash
git clone https://github.com/SwissTonyStark/GameMindsDT.git
cd GameMindsDT
pip install -r requirements.txt
```
### Preliminary experiments
Before we deep dive into our scratch decision transformer, we wanted to verify that the DT algorithm outperforms some classic RL algorithm. So the main objective of these experiments is to **verify the power of the DT algorithm in comparison to other classical algorithms**. Note that the DT used on this short experiments is an already implemented one found in the d3rlpy library. Visit the [Preliminars experiments README](experiments/README.md)

![key_to_door](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/d05b3385-73c9-4abb-a1d0-c03e06b9646f)

*The agent is trying to get the key to unlock the door to find the next level.*
***
### MineRL: Exploration and Objectives
<!-- Estic muntant a draw.io una llista d'algorismes i environments -->

![image](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/d75127fd-e5a2-4f32-b0a4-2b6ec87778a8)


<!-- EDGAR -->

We have demonstrated in previous experiments that Decision Transformers can solve games and benchmark environments such as mujoco, atari, and minigrid. However, we would really like to know if DTs can be used in more complex real-world applications. Lacking data, we have decided to use Minecraft, which, despite being a game, is an environment several orders of magnitude more complex than any of the previously proposed ones. The idea is to check if it can learn anything from human video demonstrations.

![hole_in_one](https://github.com/SwissTonyStark/GameMindsDT/assets/155813568/d6d4af15-0e29-4013-a6bb-d38f74e09921)

For an in-depth look, visit the [MineRL README](dt-mine-rl-project/README.md).

<!-- AFEGIR OMAR I SHUANG EL VOSTRE -->

<!-- ALEX?? FINALMENT HI HA README? -->

***
# Docker Integration
## Introduction to Docker

Docker played a crucial role in ensuring a consistent development environment across our team. Is an *open-source containerization platform* that enables developers to **package applications into containers—standardized executable components** combining application source code with the operating system (OS) libraries and dependencies required to run that code in any environment.

![docker logo](https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/c7ffea0a-fac7-4b4b-9e8d-bc1104028fe1)

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
Then will you have installed the base image to run our project in a container! But keep reading, you will need to add some extensions,
new images which contain more dependencies.

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
Once you have built the image you need, you can run a container based on that image. Make sure to replace `nvidia-pytorch:base`, `minerl-dt`, or `pybullet-dt` with the name of the image you have built.
You will find your images in Docker Desktop, it's recommendable to check if you have them all, but you won't need to run them manually. Visual Studio Code does it for you, it runs an instance of them.

#### Setting Up Development Environment in VSCode
After building the images, open the project in Visual Studio Code. Use the "Reopen in Container" feature from the top menu. Select the desired container from the available options.

The Docker containers are defined in the `./devcontainer` directory, where you can find their configurations.

That's it! You are now ready to work with Docker and use the built images for your projects with MineRL and PyBullet.
## Resources
Overview of our repository structure and data flow diagrams to navigate through our project's architecture efficiently.

<!-- ADJUNTAR DIAGRAMA FLOWS DE LES CARPETES? -->

## Conclusions
Summarizing our journey, achievements, challenges faced, and the insights gained through the development of GameMindsDT.
DT s'ha adaptat be a l'Atari, hem estat optimistes.., referenciar els problemes que hem tingut, retards, solucions/parches, 

**EDGAR**: In this project, I have learned that decision transformers and their variants are currently an active area of research. This is both good and bad. On one hand, it means that we can obtain promising results, but on the other, we have encountered a lack of documentation and real working examples. Many algorithms have only been theoretically tested in simple environments. Ultimately, the project has been an R&D exercise in itself. Despite the difficulties, the fact that we managed to get several environments working has been very rewarding and enriching. For my part, I plan to closely follow this technology to try to apply it in future projects.

**POL**: During my work on the GameMindsDT project, I've really gotten to grips with Decision Transformers, Reinforcement Learning, and Docker. Exploring Decision Transformers and Reinforcement Learning has opened my eyes to the powerful ways AI can solve complex problems and improve decision-making in games and simulations. At the same time, dealing with Docker taught me a lot about the importance of containerization for keeping software running smoothly across different computers. This project showed me how essential both advanced AI concepts and practical software tools are for creating effective and reliable applications. It's made me even more excited to keep working in this field, combining deep AI knowledge with solid software development skills.

**FEM CADASCÚ 3,4,5 linies de conclusions personals, que hem apres, problemes que ha tingut**
<!-- IMPORTANT OMAR WAND.DB??? QUE ESTIGUI VISUAL I BEN EXPLICAT -->


## License
This project is licensed under the [MIT License](LICENSE).

<!-- PART ANTIGA PER APROFITAR -->
<!-- A partir d'aqui esta el readme tal qual anterior -->
<!-- La idea es comparar amb la nova versio i acabar-ho de completar tot -->
<!-- Estic reorganitzant tot -->


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
