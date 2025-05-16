
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch
from torch.distributions import Categorical


class MultiOutputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sl_head = None
        self.tp_head = None

    def _build(self, lr_schedule):
        super()._build(lr_schedule)

        # Build SL/TP prediction heads after extractor is available
        self.sl_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
            nn.Sigmoid()
        )
        self.tp_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        action_logits = self.action_net(latent_pi)
        sl = torch.clamp(self.sl_head(latent_pi), 0.002, 0.05)
        tp = torch.clamp(self.tp_head(latent_pi), 0.004, 0.10)

        dist = Categorical(logits=action_logits)
        action = dist.probs.argmax(dim=1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        if torch.isnan(action_logits).any():
            print("❌ NaN in action logits")
        if torch.isnan(sl).any() or torch.isnan(tp).any():
            print("❌ NaN in SL/TP outputs")

        return action, log_prob, self.value_net(latent_vf), sl, tp

    def _predict(self, obs, deterministic=False):
        action, _, _, _, _ = self.forward(obs, deterministic)
        return action

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_net(latent_vf)

        return value, log_prob, entropy
