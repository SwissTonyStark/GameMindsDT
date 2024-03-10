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


