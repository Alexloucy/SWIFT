import os
import time as _time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .dino_utils import create_linear_input
from torch.nn import functional as F
from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from loss.make_loss import make_loss


class BottleneckAdapter(nn.Module):
    def __init__(self, channel_in, reduction=4, init_scale=0.1):
        super().__init__()
        self.down = nn.Linear(channel_in, channel_in // reduction, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.up = nn.Linear(channel_in // reduction, channel_in, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.scale * self.up(self.drop(self.act(self.down(x))))


class CustomDino(nn.Module):
    def __init__(self, cfg, num_classes, dino_model, domains):
        super().__init__()
        self.cfg = cfg
        self.dino_model = dino_model

        output_dim = self.dino_model.embed_dim
        self.adapter_dict = nn.ModuleDict()
        self.classifier_dict = nn.ModuleDict()

        self.day_night_adapter = cfg.MODEL.Day_Night_Adapter

        self.dino_layer_used = 4

        for i, domain in enumerate(domains):
            if self.day_night_adapter:
                self.adapter_dict[f"adapter_{i}"] = BottleneckAdapter(output_dim * self.dino_layer_used, 4)       # shared across day/night
                self.adapter_dict[f"day_delta_{i}"] = BottleneckAdapter(output_dim * self.dino_layer_used, 8, init_scale=0.1)   # day-specific correction
                self.adapter_dict[f"night_delta_{i}"] = BottleneckAdapter(output_dim * self.dino_layer_used, 8, init_scale=0.1) # night-specific correction
            else:
                self.adapter_dict[f"adapter_{i}"] = BottleneckAdapter(output_dim * self.dino_layer_used, 4)
            self.classifier_dict[f"classifier_{i}"] = nn.Linear(output_dim * self.dino_layer_used, num_classes[i])
        self.num_classes = num_classes
        self.domains = domains
    
    def get_classifier_scores(self, features, domain_id):
        classifier = self.classifier_dict[f"classifier_{domain_id}"]
        cls_scores = classifier(features)
        return cls_scores

    def forward(self, image, domains, time):
        # Get cls token from DINO
        x_tokens_list = self.dino_model.get_intermediate_layers(image, n=self.dino_layer_used, return_class_token=True)
        image_features = create_linear_input(x_tokens_list, self.dino_layer_used, False)
        base_features = image_features.float()

        # Check if all samples in batch have the same domain (training case)
        unique_domains = list(set(domains))
        single_domain_batch = len(unique_domains) == 1

        mixed_features = base_features.clone()
        cls_scores = None

        time = torch.tensor(time).to(image.device) if time is not None else None
        domains = torch.tensor(domains).to(image.device)

        if self.day_night_adapter:
            if time is None:
                raise ValueError("Time information (day/night) must be provided when using day/night adapters.")

            time_domain_pairs = torch.stack((time, domains), dim=1)
            unique_pairs = torch.unique(time_domain_pairs, dim=0)

            # Apply shared adapter then time-specific delta
            for p in unique_pairs.tolist():
                t, d = p
                shared_adapter = self.adapter_dict[f"adapter_{d}"]
                day_delta = self.adapter_dict[f"day_delta_{d}"]
                night_delta = self.adapter_dict[f"night_delta_{d}"]
                mask = ((time == t) & (domains == d))
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                sub_base_features = base_features.index_select(0, idx)
                shared_out = shared_adapter(sub_base_features)
                if t == 1:  # day
                    mixed_features[idx] = day_delta(shared_out)
                else:  # night
                    mixed_features[idx] = night_delta(shared_out)

        else:
            for domain_id in unique_domains:
                idx = (domains == domain_id).nonzero(as_tuple=False).squeeze(1)
                sub_base_features = base_features.index_select(0, idx)
                adapter = self.adapter_dict[f"adapter_{domain_id}"]
                mixed_features[idx] = adapter(sub_base_features)
        
        if single_domain_batch:
            domain_id = unique_domains[0]
            cls_scores = self.get_classifier_scores(mixed_features, domain_id)

        mixed_features_norm = F.normalize(mixed_features, dim=-1, eps=1e-6)
        return cls_scores, mixed_features_norm


@MODEL_REGISTRY.register()
class Adapter(Trainer):
    """
    Dino-Adapter for multi-source domain adaptation
    """

    def build_model(self):
        print("Loading Dino backbone: {}".format(self.cfg.MODEL.Adapter.BACKBONE))
        dino_model = torch.hub.load(self.cfg.MODEL.Adapter.REPO, self.cfg.MODEL.Adapter.BACKBONE, source='local',
                                    weights=self.cfg.MODEL.Adapter.WEIGHT_PATH)
        self.dino_model = dino_model.to(self.device)
        self.dino_model.eval()

        print("Building Custom Dino")
        self.model = CustomDino(
            self.cfg, self.data_manager.num_classes, dino_model, self.data_manager.get_target_domains
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter_dict" not in name and "classifier_dict" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # Mixed precision setup (defaults to on unless explicitly disabled in config)
        self.use_mixed_precision = getattr(self.cfg.TRAIN, "MIXED_PRECISION", True)
        dtype_cfg = getattr(self.cfg.TRAIN, "MIXED_PRECISION_DTYPE", "auto")
        self.amp_enabled = self.use_mixed_precision and torch.cuda.is_available()
        if not self.amp_enabled:
            self.amp_dtype = torch.float32
        elif isinstance(dtype_cfg, torch.dtype):
            self.amp_dtype = dtype_cfg
        else:
            dtype_str = str(dtype_cfg).lower()
            if dtype_str == "auto":
                self.amp_dtype = torch.bfloat16 if self._bf16_supported() else torch.float16
            elif "bf16" in dtype_str or "bfloat16" in dtype_str:
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        print(f"Using AMP: {self.amp_enabled}, dtype: {self.amp_dtype}")
        self.domain_scalers = {}

        # Create domain-specific optimizers and schedulers
        self.domain_optimizers = {}
        self.domain_schedulers = {}
        
        # use target domains which is a subset of source domains
        for domain_id, domain_name in enumerate(self.data_manager.get_target_domains):
            # Get domain-specific parameters (adapter + classifier for this domain)
            domain_params = []
            if self.cfg.MODEL.Day_Night_Adapter:
                domain_params.extend(list(self.model.adapter_dict[f"adapter_{domain_id}"].parameters()))
                domain_params.extend(list(self.model.adapter_dict[f"day_delta_{domain_id}"].parameters()))
                domain_params.extend(list(self.model.adapter_dict[f"night_delta_{domain_id}"].parameters()))
            else:
                domain_params.extend(list(self.model.adapter_dict[f"adapter_{domain_id}"].parameters()))
            domain_params.extend(list(self.model.classifier_dict[f"classifier_{domain_id}"].parameters()))

            # Get domain-specific learning rate multiplier if available
            lr_multiplier = 1.0
            if hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_LR_MULTIPLIERS'):
                multipliers = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_LR_MULTIPLIERS
                if isinstance(multipliers, (list, tuple)) and len(multipliers) > domain_id:
                    lr_multiplier = multipliers[domain_id]
            
            # Create domain-specific optimizer with custom learning rate
            domain_optimizer = build_optimizer(domain_params, self.cfg.OPTIM, lr_multiplier=lr_multiplier)
            self.domain_optimizers[domain_id] = domain_optimizer
            
            # Create domain-specific scheduler with custom config if available
            domain_scheduler = self._build_domain_scheduler(domain_optimizer, domain_id, len(self.data_manager.get_target_domains))
            self.domain_schedulers[domain_id] = domain_scheduler
            
            # Register each domain optimizer and scheduler
            self.model_registeration(
                f"dino_adapter_domain_{domain_id}",
                self.model,
                domain_optimizer,
                domain_scheduler,
            )
            self.domain_scalers[domain_id] = GradScaler(enabled=self.amp_enabled)
            
            # Print domain-specific configuration info
            scheduler_type = "default"
            if (hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and
                hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_SCHEDULERS') and
                len(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS) > domain_id):
                scheduler_type = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS[domain_id]
            print(f"Domain {domain_id}: LR base={self.cfg.OPTIM.LR:.6f}, mult={lr_multiplier:.3f}, Scheduler={scheduler_type}")
            

    def save_model(self, current_epoch, save_dir, model_name="name"):
        model_names = self.get_model_names()
        for mn in model_names:
            full_dict = self._models[mn].state_dict()
            # Exclude frozen backbone weights — only keep adapter/classifier weights
            model_dict = {k: v for k, v in full_dict.items() if not k.startswith("dino_model.")}

            optimizer_dict = None
            if self._optimizers[mn] is not None:
                optimizer_dict = self._optimizers[mn].state_dict()

            lr_scheduler_state_dict = None
            if self._lr_schedulers[mn] is not None:
                lr_scheduler_state_dict = self._lr_schedulers[mn].state_dict()

            new_model_dict = OrderedDict()
            for key, value in model_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                new_model_dict[key] = value

            model_domains = self.cfg.DATASET.TARGET_DOMAINS if hasattr(self.cfg.DATASET, 'TARGET_DOMAINS') else []
            if len(model_domains) > 0:
                model_domains = "-".join(model_domains)
            current_time = _time.strftime("%Y-%m-%d_%H-%M-%S", _time.localtime())
            model_full_name = f"{self.cfg.MODEL.NAME}_domains_{model_domains}_{current_time}.pth.tar"

            fpath = os.path.join(save_dir, model_full_name + str(current_epoch + 1))
            torch.save(
                {
                    "state_dict": new_model_dict,
                    "epoch": current_epoch + 1,
                    "optimizer": optimizer_dict,
                    "lr_scheduler": lr_scheduler_state_dict,
                },
                fpath,
            )
            print("Model Saved to: {}".format(fpath))

    def forward_backward(self, batch_data):
        image, target, domains, time = self.parse_batch_train(batch_data)

        # all samples in the batch have the same domain
        domain = domains[0]
        optimizer_name = f"dino_adapter_domain_{domain}"
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            output, feat = self.model(image, domains, time)
            loss_func, center_criterion = make_loss(self.cfg, self.num_classes[domain], self.device)
            loss, ID_LOSS, TRI_LOSS = loss_func(output, feat, target, None)

        if self.amp_enabled and self.amp_dtype == torch.float16:
            optimizer = self._optimizers[optimizer_name]
            optimizer.zero_grad()
            self.detect_abnormal_loss(loss)
            scaler = self.domain_scalers[domain]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Use domain-specific optimizer for backward and update
            self.model_backward_and_update(loss, optimizer_name)

        loss_summary = {
            "domain": self.data_manager.get_source_domains[domain],
            "domain_id": domain,
            "loss": loss.item(),
            "ID_LOSS": ID_LOSS,
            "TRI_LOSS": TRI_LOSS,
            "acc": compute_accuracy(output, target)[0].item(),
        }

        # LR scheduler now steps once per epoch in Trainer.after_epoch

        return loss_summary

    def model_inference(self, batch_data, domains, time=None):
        if self.cfg.MODEL.Day_Night_Adapter and time is None:
            raise ValueError("Time labels are required when Day_Night_Adapter is enabled.")
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            _, feat = self.model(batch_data, domains, time)
        return feat

    def _build_domain_scheduler(self, optimizer, domain_id, num_domains):
        """Build domain-specific learning rate scheduler with custom config if available."""
        # Check if domain-specific scheduler config exists
        if hasattr(self.cfg.OPTIM, 'DOMAIN_OPTIM') and hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_SCHEDULERS'):
            domain_schedulers = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_SCHEDULERS
            if isinstance(domain_schedulers, (list, tuple)) and len(domain_schedulers) == num_domains:
                scheduler_type = domain_schedulers[domain_id]
                
                # Get domain-specific step size if using StepLR
                step_size = None
                if (scheduler_type == "StepLR" and 
                    hasattr(self.cfg.OPTIM.DOMAIN_OPTIM, 'DOMAIN_STEP_SIZES') and
                    isinstance(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES, (list, tuple)) and
                    len(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES) > domain_id):
                    step_size = self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_STEP_SIZES[domain_id]
                
                return build_lr_scheduler(optimizer, self.cfg.OPTIM, 
                                        scheduler_type=scheduler_type, step_size=step_size)
        
        # Fall back to default scheduler
        return build_lr_scheduler(optimizer, self.cfg.OPTIM)

    def _bf16_supported(self):
        """Detect whether the current CUDA device supports bf16."""
        if not torch.cuda.is_available():
            return False
        has_is_bf16 = hasattr(torch.cuda, "is_bf16_supported")
        if has_is_bf16:
            try:
                return torch.cuda.is_bf16_supported()
            except RuntimeError:
                return False
        major, _ = torch.cuda.get_device_capability(self.device)
        return major >= 8
