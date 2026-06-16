import os

from datasets.base_dataset import BaseImageDataset
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Cat(BaseImageDataset):
    def __init__(self, cfg, domain_label, verbose=True):
        dataset_dir = "Cat"
        root = cfg.DATASET.ROOT
        dataset_path = os.path.join(root, dataset_dir)
        
        domain = "cat"
        
        train_dir = os.path.join(dataset_path, "train")
        gallery_dir = os.path.join(dataset_path, "gallery")
        query_dir = os.path.join(dataset_path, "query")
        
        need_day_night = cfg.MODEL.Day_Night_Adapter if hasattr(cfg.MODEL, 'Day_Night_Adapter') else False

        super().__init__(
            cfg=cfg,
            dataset_dir=dataset_path,
            domain=domain,
            domain_label=domain_label,
            train_dir=train_dir,
            gallery_dir=gallery_dir,
            query_dir=query_dir,
            need_day_night=need_day_night,
            verbose=verbose
        )
