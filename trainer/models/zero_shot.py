import torch
from .dino_utils import create_linear_input

from trainer import MODEL_REGISTRY, Trainer


@MODEL_REGISTRY.register()
class ZeroShot(Trainer):
    def build_model(self):
        model_cfg = self.cfg.MODEL.ZeroShot
        self.model = torch.hub.load(
            model_cfg.REPO,
            model_cfg.BACKBONE,
            source="local",
            weights=model_cfg.WEIGHT_PATH,
        )
        self.model.eval()
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model = self.model.to(self.device)

    def model_inference(self, input_data, domain, time=None):
        x_tokens_list = self.model.get_intermediate_layers(
            input_data, n=1, return_class_token=True
        )
        image_features = create_linear_input(x_tokens_list, 1, False)
        image_features = torch.nn.functional.normalize(image_features, dim=-1, eps=1e-6)
        return image_features
