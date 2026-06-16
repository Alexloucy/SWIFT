import os

from datasets.base_dataset import BaseImageDataset
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FeralPig(BaseImageDataset):
    """
    Dataset statistics:
        - images: 818 (train) + 152 (gallery) + 115 (query)
        - identities: 89 (train) + 21 (gallery) + 21 (query)
    """
    def __init__(self, cfg, domain_label, verbose=True):
        dataset_dir = "feralPig"
        root = cfg.DATASET.ROOT
        dataset_path = os.path.join(root, dataset_dir)
        
        domain = "feralpig"
        
        train_dir = os.path.join(dataset_path, "train")
        gallery_dir = os.path.join(dataset_path, "gallery")
        query_dir = os.path.join(dataset_path, "query")
        
        super().__init__(
            cfg=cfg,
            dataset_dir=dataset_path,
            domain=domain,
            domain_label=domain_label,
            train_dir=train_dir,
            gallery_dir=gallery_dir,
            query_dir=query_dir,
            relabel_train=True,
            verbose=verbose
        )
