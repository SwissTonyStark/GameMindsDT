# GameMindsDT - Project Overview

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/232c06e1-7cb4-4d0b-b2ae-11d24c79a2ef" alt="GameMindstDT Atari & Minecraft">
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

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/a9e6e770-8ec6-4e02-a5be-fab2d814f058" alt="Gantt Chart">
</p>

# State of the Art: Decision Transformer 🌐
The cornerstone of our project is **Reinforcement Learning**, specifically focusing on offline RL and data sequences rather than direct environment interaction. This project delves into the architecture of the **Decision Transformer**, reducing the RL problem to a conditional sequence modeling problem.

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/10a57bff-849c-4d44-985b-4f72c83c8d03" alt="Decision Transformer">
</p>

# Environments 🎮
We explored a variety of algorithms and environments, ranging from OpenAI Gym to more complex simulations, each presenting unique challenges and learning opportunities.

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/4e11aaf8-9ae9-4721-841e-db724e772de4" alt="Environments GameMinds DT">
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
We trained models on Atari games using a Decision Transformer to benchmark our approach against this visually rich and diverse environment, specially challenging due to the difficulty of credit assignment arising from the delay between actions and resulting rewards.
For a detailed overview, refer to our [DT-Atari README](DT-atari/README.md).

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/01c5365c-5b41-4ac1-8ca2-96e37aa74294" alt="Atari Games">
</p>

*Atari games provide a rich platform for demonstrating the Decision Transformer's capabilities.*

## MineRL: Exploration and Objectives ⛏️
Pushing the boundaries further, we applied the Decision Transformer to Minecraft via the MineRL environment. This experiment aimed to explore the model's potential in complex, real-world tasks based on human video demonstrations. Dive deeper into our findings in the [MineRL README](dt-mine-rl-project/README.md).

<p align="center">
  <img src="https://github.com/SwissTonyStark/GameMindsDT/assets/146961986/d75127fd-e5a2-4f32-b0a4-2b6ec87778a8" alt="MineRL Environment">
</p>

*Minecraft's complexity presents a significant challenge, testing the Decision Transformer's learning efficiency and adaptability.*

## Docker Integration 🐳
## Introduction to Docker

Docker played a crucial role in ensuring a consistent development environment across our team. It's an *open-source containerization platform* that enables developers to **package applications into containers—standardized executable components** combining application source code with the operating system (OS) libraries and dependencies required to run that code in any environment.

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

<!-- ADJUNTAR DIAGRAMA FLOWS DE LES CARPETES? -->

## Conclusions
Summarizing our journey, achievements, challenges faced, and the insights gained through the development of GameMindsDT.
DT s'ha adaptat be a l'Atari, hem estat optimistes.., referenciar els problemes que hem tingut, retards, solucions/parches, 

**EDGAR**: In this project, I have learned that decision transformers and their variants are currently an active area of research. This is both good and bad. On one hand, it means that we can obtain promising results, but on the other, we have encountered a lack of documentation and real working examples. Many algorithms have only been theoretically tested in simple environments. Ultimately, the project has been an R&D exercise in itself. Despite the difficulties, the fact that we managed to get several environments working has been very rewarding and enriching. For my part, I plan to closely follow this technology to try to apply it in future projects.

**POL**: During my work on the GameMindsDT project, I've really gotten to grips with Decision Transformers, Reinforcement Learning, and Docker. Exploring Decision Transformers and Reinforcement Learning has opened my eyes to the powerful ways AI can solve complex problems and improve decision-making in games and simulations. At the same time, dealing with Docker taught me a lot about the importance of containerization for keeping software running smoothly across different computers. This project showed me how essential both advanced AI concepts and practical software tools are for creating effective and reliable applications. It's made me even more excited to keep working in this field, combining deep AI knowledge with solid software development skills.

**Shuang**: Concerning reinforcement learning, it is widely recognized as an emerging field of research with limited real-world applications beyond video games. Throughout the project, we have witnessed the potential of decision transformers, particularly in scenarios where a virtual environment is unavailable and only a dataset is provided. A notable drawback is that decision transformers, similar to all transformers, require a substantial dataset that is also well-organized. During our experiments, we observed promising results in some environments, but in others, the outcomes were not as favorable. Several factors could contribute to this, including insufficient data, an incomplete exploration of all hyperparameter options, etc. For future works, I would improve the already done projects and expand it.  

**FEM CADASCÚ 3,4,5 linies de conclusions personals, que hem apres, problemes que ha tingut**
<!-- IMPORTANT OMAR WAND.DB??? QUE ESTIGUI VISUAL I BEN EXPLICAT -->


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
