
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class MultiOutputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)

class MultiOutputPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MultiOutputExtractor,
            **kwargs,
        )
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.sl_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        self.tp_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob  # ✅ Must match SB3 expected output

    def predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions

    def predict_sl_tp(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        sl = self.sl_head(latent_pi)
        tp = self.tp_head(latent_pi)
        return sl, tp
