from __future__ import annotations

from .feature_extractor_torchhub import TorchHubGlobalFeatureExtractor


class SALADFeatureExtractor(TorchHubGlobalFeatureExtractor):
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        hub_model: str = "dinov2_salad",
        pretrained: bool = True,
        image_size: int | tuple[int, int] = (322, 322),
    ):
        # The official Torch Hub entrypoint `dinov2_salad` loads the released
        # SALAD checkpoint and returns an 8448D global descriptor.
        dim = 8448 if hub_model == "dinov2_salad" else None
        super().__init__(
            repo="serizba/salad",
            hub_model=hub_model,
            hub_kwargs={"backbone": backbone, "pretrained": pretrained},
            dim=dim,
            resize=image_size,
            interpolation="bicubic",
            allow_mps=False,
            mps_disabled_reason=(
                "SALAD falls back from MPS because the DINOv2 positional "
                "interpolation path uses bicubic upsampling, which is not "
                "implemented on MPS in the current PyTorch release"
            ),
            missing_dependency_message=(
                "SALAD requires extra dependencies from the official implementation. "
                "Install them with: "
                "`python -m pip install pytorch-lightning==2.4.0 "
                "torchmetrics==1.4.3 pytorch-metric-learning==2.8.1 "
                "prettytable==3.16.0 pandas==2.2.3`"
            ),
        )
