import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt

import gym
import d4rl_atari

def evaluate_on_env(model, env, hparams, device, num_eval_ep=10, max_test_ep_len=8000,
                    render=False, temperature=1.0, draw_state=False):

    rtg_target = hparams['target_reward']
    context_len = hparams['context_length']
    state_dim = hparams['state_dim']

    model.eval()
    result_rewards = []
    result_ep_len = []
    frames = []

    env.reset()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            total_reward = 0
            total_timesteps = 0

            # same as timesteps used for training the transformer
            timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
            timesteps = timesteps.repeat(1, 1).to(device)

            # zeros place holders
            actions = torch.zeros((1, max_test_ep_len, 1), dtype=torch.float32, device=device) # (B, T, act_dim=1)
            states = torch.zeros((1, max_test_ep_len, state_dim), dtype=torch.float32, device=device) # (B, T, 4*84*84          
            rewards_to_go = torch.zeros((1, max_test_ep_len, 1),  dtype=torch.float32, device=device) # (B, T, 1)
                               
            # init episode
            running_state = env.reset()

            running_reward = 0
            running_rtg = rtg_target

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                running_state = np.array(running_state).astype(np.float32) / 255.0
                running_state = torch.from_numpy(running_state)#.unsqueeze(0)
                running_state = running_state.flatten()

                states[0, t] = running_state.to(device)

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - running_reward
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    act_preds= model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])

                    logits = act_preds[:, t, :] / temperature

                else:
                    act_preds= model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    logits = act_preds[:, -1, :].detach()

                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                act = torch.multinomial(probs, num_samples=1).item()

                running_state, running_reward, done, _ = env.step(act)

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if draw_state: # draw state images in console
                  plt.imshow(np.array(running_state[0]).astype(np.float32))
                  plt.show()
                if render: # store images in frames[]
                  frames.append(env.render(mode="rgb_array"))
                if done:
                    break

            result_rewards.append(total_reward)
            result_ep_len.append(total_timesteps)

    print('max_reward ', max(result_rewards))
    print('max_ep_lens ', max(result_ep_len))
    print('*****> rewards average ', sum(result_rewards)/num_eval_ep)
    return frames