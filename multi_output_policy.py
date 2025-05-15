
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
import torch.nn as nn
import torch
from torch.distributions import Categorical


class MultiOutputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MultiOutputPolicy, self).__init__(*args, **kwargs)

        # Custom heads for SL and TP
        self.sl_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
            nn.Sigmoid()
        )
        self.tp_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
            nn.Sigmoid()
        )

        self._build()

    def _build_mlp_extractor(self):
        self.mlp_extractor = self.mlp_extractor_class(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.n)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Core action logits
        action_logits = self.action_net(latent_pi)

        # Predict SL/TP and clip them
        sl = torch.clamp(self.sl_head(latent_pi), 0.002, 0.05)
        tp = torch.clamp(self.tp_head(latent_pi), 0.004, 0.10)

        # Build categorical action distribution
        dist = Categorical(logits=action_logits)

        # Sample or choose action
        action = dist.probs.argmax(dim=1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        if torch.isnan(action_logits).any():
            print("❌ NaN detected in action logits")
        if torch.isnan(sl).any() or torch.isnan(tp).any():
            print("❌ NaN detected in SL/TP predictions")

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


register_policy("MultiOutputPolicy", MultiOutputPolicy)
