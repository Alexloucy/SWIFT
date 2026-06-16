import os

from datasets.base_dataset import BaseImageDataset
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Skink(BaseImageDataset):
    """
    Skink Dataset
    """
    def __init__(self, cfg, domain_label, verbose=True):
        dataset_dir = "skink"
        root = cfg.DATASET.ROOT
        dataset_path = os.path.join(root, dataset_dir)

        domain = "skink"
        need_day_night = getattr(cfg.MODEL, "Day_Night_Adapter", False)

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
            need_day_night=need_day_night,
            relabel_train=True,
            verbose=verbose
        )
