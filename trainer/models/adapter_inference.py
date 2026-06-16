import os
import time as _time
from collections import OrderedDict

import torch
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None

from trainer import MODEL_REGISTRY, Trainer
from .adapter import CustomDino


@MODEL_REGISTRY.register()
class AdapterInference(Trainer):
    """
    Inference-only wrapper for Adapter checkpoints.

    Uses the identical CustomDino architecture (4-layer DINO backbone,
    shared adapter + day_delta / night_delta per domain) so that weights
    saved by Adapter load without key or shape mismatches.

    No optimisers or classifiers are created — forward/backward raises.
    """

    def build_model(self):
        adapter_cfg = self.cfg.MODEL.AdapterInference
        backbone_cfg = self.cfg.MODEL.Adapter
        dino_ckpt = backbone_cfg.WEIGHT_PATH
        adapter_ckpt = adapter_cfg.ADAPTER_WEIGHTS

        if not adapter_ckpt:
            raise ValueError(
                "cfg.MODEL.AdapterInference.ADAPTER_WEIGHTS must be set."
            )
        if not os.path.isfile(adapter_ckpt):
            raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_ckpt}")

        print(f"Loading Dino backbone: {backbone_cfg.BACKBONE}")
        dino_model = torch.hub.load(
            backbone_cfg.REPO,
            backbone_cfg.BACKBONE,
            source="local",
            weights=dino_ckpt,
        )
        dino_model = dino_model.to(self.device)
        dino_model.eval()

        print("Building CustomDino (SharedDelta architecture) for inference")
        self.model = CustomDino(
            self.cfg,
            self.data_manager.num_classes,
            dino_model,
            self.data_manager.get_target_domains,
        )
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        self._load_adapter_weights(adapter_ckpt)

        # AMP setup mirrors Adapter training.
        use_mixed_precision = getattr(self.cfg.TRAIN, "MIXED_PRECISION", True)
        self.amp_enabled = use_mixed_precision and torch.cuda.is_available()
        if not self.amp_enabled:
            self.amp_dtype = torch.float32
        else:
            dtype_cfg = str(getattr(self.cfg.TRAIN, "MIXED_PRECISION_DTYPE", "auto")).lower()
            if "bf16" in dtype_cfg or "bfloat16" in dtype_cfg:
                self.amp_dtype = torch.bfloat16
            elif "fp16" in dtype_cfg or "float16" in dtype_cfg:
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.bfloat16 if self._bf16_supported() else torch.float16
        print(f"Using AMP: {self.amp_enabled}, dtype: {self.amp_dtype}")

        # Register once with no optimizer/scheduler — needed by Trainer.test()
        self.model_registeration("dino_shareddelta_inference", self.model, None, None)

    def _load_adapter_weights(self, adapter_ckpt: str) -> None:
        if add_safe_globals is not None:
            add_safe_globals([CosineAnnealingLR])

        checkpoint = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Keep only adapter_dict keys (skip classifier_dict and dino_model)
        adapter_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("adapter_dict.")
        }

        load_result = self.model.adapter_dict.load_state_dict(
            {k[len("adapter_dict."):]: v for k, v in adapter_state.items()},
            strict=True,
        )
        print(f"Loaded adapter weights from: {adapter_ckpt}")
        if load_result.missing_keys:
            print(f"  Warning — missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Warning — unexpected keys: {load_result.unexpected_keys}")

    def forward_backward(self, batch_data):
        raise RuntimeError(
            "AdapterInference is inference-only and does not support training."
        )

    def model_inference(self, batch_data, domains, time=None):
        if self.cfg.MODEL.Day_Night_Adapter and time is None:
            raise ValueError("Time labels are required when Day_Night_Adapter is enabled.")
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            _, feat = self.model(batch_data, domains, time)
        return feat

    def _bf16_supported(self):
        if not torch.cuda.is_available():
            return False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return torch.cuda.is_bf16_supported()
            except RuntimeError:
                return False
        major, _ = torch.cuda.get_device_capability(self.device)
        return major >= 8
