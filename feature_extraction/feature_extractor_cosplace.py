from __future__ import annotations

from .feature_extractor_torchhub import TorchHubGlobalFeatureExtractor


class CosPlaceFeatureExtractor(TorchHubGlobalFeatureExtractor):
    def __init__(self, backbone: str = "ResNet50", fc_output_dim: int = 2048):
        super().__init__(
            repo="gmberton/cosplace",
            hub_model="get_trained_model",
            hub_kwargs={"backbone": backbone, "fc_output_dim": fc_output_dim},
            dim=fc_output_dim,
            resize=480,
        )
