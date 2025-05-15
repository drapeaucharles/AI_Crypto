
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class MultiOutputFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(self.flatten(observations))

class MultiOutputPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MultiOutputFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs,
        )
        # Add SL/TP heads
        self.sl_head = nn.Sequential(nn.Linear(self.mlp_extractor.latent_pi.shape[1], 1), nn.Sigmoid())
        self.tp_head = nn.Sequential(nn.Linear(self.mlp_extractor.latent_pi.shape[1], 1), nn.Sigmoid())

    def forward(self, obs, deterministic=False):
        actions, value, log_prob = super().forward(obs, deterministic)
        sl = self.sl_head(self.mlp_extractor.latent_pi)
        tp = self.tp_head(self.mlp_extractor.latent_pi)
        return actions, value, log_prob, sl, tp

    def _predict(self, observation, deterministic=False):
        actions = super()._predict(observation, deterministic)
        sl = self.sl_head(self.mlp_extractor.latent_pi)
        tp = self.tp_head(self.mlp_extractor.latent_pi)
        return actions, sl, tp
