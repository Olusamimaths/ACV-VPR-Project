from __future__ import annotations

from .feature_extractor_torchhub import TorchHubGlobalFeatureExtractor


class SelaVPRPlusPlusFeatureExtractor(TorchHubGlobalFeatureExtractor):
    def __init__(
        self,
        backbone: str = "dinov2-base",
        aggregation: str = "gem",
        image_size: int | tuple[int, int] = (518, 518),
    ):
        if backbone not in {"dinov2-base", "dinov2-large"}:
            raise ValueError(f"Unsupported SelaVPR++ backbone: {backbone}")
        if aggregation not in {"gem", "boq", "salad"}:
            raise ValueError(f"Unsupported SelaVPR++ aggregation: {aggregation}")

        if aggregation == "gem":
            dim = 2048 if backbone == "dinov2-base" else 4096
        elif aggregation == "boq":
            dim = 12288
        else:
            dim = 8448

        super().__init__(
            repo="Lu-Feng/SelaVPRplusplus",
            hub_model="SelaVPRplusplus",
            hub_kwargs={
                "backbone": backbone,
                "aggregation": aggregation,
                "hashing": False,
                "rerank": False,
            },
            dim=dim,
            resize=image_size,
            interpolation="bicubic",
            small_batch_size=2,
            large_batch_size=2,
            allow_mps=False,
            mps_disabled_reason=(
                "SelaVPR++ falls back from MPS because its DINOv2 positional "
                "interpolation path uses bicubic upsampling, which is not "
                "implemented on MPS in the current PyTorch release"
            ),
            unwrap_dataparallel=True,
            missing_dependency_message=(
                "SelaVPR++ could not be loaded from Torch Hub. "
                "Please ensure your PyTorch environment is installed correctly, "
                "then retry so Torch Hub can download the official model files."
            ),
        )
