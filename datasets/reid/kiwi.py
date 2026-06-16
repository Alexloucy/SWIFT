import os

from datasets.base_dataset import BaseImageDataset
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Kiwi(BaseImageDataset):
    """
    Dataset statistics:
        - identities: 38 (train) + 5 (gallery) + 5 (query)
        - images: 510 (train) + 25 (gallery) + 25 (query)
    """
    def __init__(self, cfg, domain_label, verbose=True):
        dataset_dir = "kiwi"
        root = cfg.DATASET.ROOT
        dataset_path = os.path.join(root, dataset_dir)
        
        domain = "kiwi"
        
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
