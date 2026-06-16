import os

from datasets.base_dataset import BaseImageDataset
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Cattle(BaseImageDataset):
    """
    Reference: W. Andrew, C. Greatwood and T. Burghardt, "Visual Localisation and Individual Identification of Holstein Friesian Cattle via Deep Learning," 
                2017 IEEE International Conference on Computer Vision Workshops (ICCVW), Venice, Italy, 2017, pp. 2850-2859, doi: 10.1109/ICCVW.2017.336.
    """
    def __init__(self, cfg, domain_label, verbose=True):
        dataset_dir = "FriesianCattle2017"
        root = cfg.DATASET.ROOT
        dataset_path = os.path.join(root, dataset_dir)
        
        domain = "cattle"
        
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
            relabel_train=False,
            verbose=verbose
        )
