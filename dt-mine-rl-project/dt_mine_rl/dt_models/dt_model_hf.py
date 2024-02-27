from transformers import DecisionTransformerModel
import torch

from dataclasses import dataclass

from dt_models.dt_model_common import ActionPolicy, AgentDT, GlobalPositionEncoding, MultiCategorical
from lib.common import AGENT_DT_ACTION_DIM, AGENT_DT_NUM_CAMERA_ACTIONS, AGENT_DT_NUN_ESC_BUTTON
from transformers import DecisionTransformerConfig
class TrainableDTHF(AgentDT,DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_timestep = GlobalPositionEncoding(
            config.hidden_size, config.max_ep_len + 1, config.n_positions // 3
        )

        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        
        self.agent_num_button_actions = config.agent_num_button_actions
        self.agent_num_camera_actions = config.agent_num_camera_actions
        self.agent_esc_button = config.agent_esc_button

        self.temperature_buttons = 1
        self.temperature_camera = 1
        self.temperature_esc = 1

        self.disable_esc_button = False

        self.predict_action = ActionPolicy(config)

    def forward(self, **kwargs):
        del kwargs["return_loss"]
        output = super().forward(**kwargs)
        action_targets = kwargs.get("actions")
        attention_mask = kwargs.get("attention_mask")

        if 'return_dict' in kwargs:
            action_preds = output.action_preds
        else:
            action_preds = output[1]

        action_preds_button = action_preds["buttons"] * attention_mask.float().unsqueeze(-1)
        action_preds_camera = action_preds["camera"] * attention_mask.float().unsqueeze(-1)
        action_preds_esc = action_preds["esc"] * attention_mask.float().unsqueeze(-1)

        multi_categorical = MultiCategorical(action_preds_button, action_preds_camera, action_preds_esc)

        action_targets = action_targets * attention_mask.float().unsqueeze(-1)
        action_targets_button = action_targets[:, :, 0].long()
        action_targets_camera = action_targets[:, :, 1].long()
        action_targets_esc = action_targets[:, :, 2].long()
 
        log_probs = multi_categorical.log_prob(action_targets_button, action_targets_camera, action_targets_esc)

        loss = -log_probs.mean()

        return {"loss": loss}

    @torch.no_grad()
    def original_forward(self, **kwargs):

        output = super().forward(**kwargs)
            
        return output

    def get_dt_action(self, states, actions, rewards, returns_to_go, timesteps, device, temperature_camera=None):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        positions = self.config.n_positions // 3

        states = states[:, -positions :]
        actions = actions[:, -positions :]
        returns_to_go = returns_to_go[:, -positions :]
        timesteps = timesteps[:, -positions :]
        padding = positions - states.shape[1]
        padding_actions = positions - actions.shape[1]

        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding, device=device), torch.ones(states.shape[1], device=device)])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding ,self.state_dim), device=device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding_actions, self.act_dim), device=device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=device), returns_to_go], dim=1).float()

        timesteps = torch.cat([torch.zeros((1, padding), device=device, dtype=torch.long), timesteps], dim=1)

        state_preds, action_logits, return_preds = self.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        action_logits_button = action_logits["buttons"][:,-1]
        action_logits_camera = action_logits["camera"][:,-1]
        action_logits_esc = action_logits["esc"][:,-1]

        multi_categorical = MultiCategorical(action_logits_button, action_logits_camera, action_logits_esc)

        if temperature_camera is None:
            temperature_camera = self.temperature_camera
        np_actions = multi_categorical.sample(self.temperature_buttons, temperature_camera, self.temperature_esc)

        actions = torch.tensor(np_actions, device=device, dtype=torch.long)

        action_preds_button = actions[0].unsqueeze(dim=0)
        action_preds_camera = actions[1].unsqueeze(dim=0)
        action_preds_esc = actions[2].unsqueeze(dim=0)

        if self.disable_esc_button:
            action_preds_esc = torch.zeros(1, device=device, dtype=torch.long)

        action_preds = torch.stack([action_preds_button, action_preds_camera, action_preds_esc], dim=-1)

        return action_preds

    @staticmethod
    def from_config(**config) -> AgentDT:
        config = DecisionTransformerConfig(
            n_head=config["n_heads"],
            n_layer=config["n_layers"],
            hidden_size=config["hidden_size"],
            n_positions=config["sequence_length"] * 3,
            max_ep_len=config["max_ep_len"],
            state_dim=config["embedding_dim"], 
            act_dim=AGENT_DT_ACTION_DIM,
            agent_num_button_actions = config["button_encoder_num_actions"],
            agent_num_camera_actions = AGENT_DT_NUM_CAMERA_ACTIONS,
            agent_esc_button = AGENT_DT_NUN_ESC_BUTTON,
            temperature_button = config["temperature_buttons"],
            temperature_camera = config["temperature_camera"],
            temperature_esc = config["temperature_esc"]
        )
        agent = TrainableDTHF(config)

        return agent

    @staticmethod
    def from_saved(**config)->'TrainableDTHF':
        return TrainableDTHF.from_pretrained(config["models_dir"])