from collections import defaultdict
import os
import shutil
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tabulate import tabulate
from .build_dataset import build_dataset
from .sampler import build_sampler
from .transforms import build_transform
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_collate_fn(batch):
    """Collate function for training data loader."""
    batch_dict = {}
    img_paths, aids, camids, viewids, imgs, domains, time = [], [], [], [], [], [], []
    for b_dict in batch:
        img_paths.append(b_dict["img_path"])
        aids.append(b_dict["aid"])
        camids.append(b_dict["camid"])
        viewids.append(b_dict["viewid"])
        imgs.append(b_dict["img"])
        domains.append(b_dict["domain"]) 
        time.append(b_dict["time"])  
    
    batch_dict["img_paths"] = img_paths
    batch_dict["aids"] = torch.tensor(aids, dtype = torch.int64)
    batch_dict["camids"] = camids
    batch_dict["viewids"] = torch.tensor(viewids, dtype = torch.int64)
    batch_dict["imgs"] = torch.stack(imgs, dim = 0)
    batch_dict["domains"] = domains 
    batch_dict["time"] = time

    return batch_dict


def test_collate_fn(batch):
    """Collate function for test data loader."""
    batch_dict = {}
    img_paths, aids, camids, viewids, imgs, domains, time = [], [], [], [], [], [], []
    for b_dict in batch:
        img_paths.append(b_dict["img_path"])
        aids.append(b_dict["aid"])
        camids.append(b_dict["camid"])
        viewids.append(b_dict["viewid"])
        imgs.append(b_dict["img"])
        domains.append(b_dict["domain"])  
        time.append(b_dict["time"])

    
    batch_dict["img_paths"] = img_paths
    batch_dict["aids"] = torch.tensor(aids, dtype = torch.int64)
    batch_dict["camids"] = camids
    batch_dict["viewids"] = torch.tensor(viewids, dtype = torch.int64)
    batch_dict["imgs"] = torch.stack(imgs, dim = 0)
    batch_dict["domains"] = domains  
    batch_dict["time"] = time

    return batch_dict


def build_data_loader(
        cfg, 
        sampler_type = "RandomIdentitySampler", 
        data_source = None, 
        batch_size = 32, 
        transform = None, 
        is_train = True,
        source_domains = None
):
    if is_train:
        need_day_night = cfg.MODEL.Day_Night_Adapter if hasattr(cfg.MODEL, 'Day_Night_Adapter') else False
        # Day/Night aware sampler only for single-domain training
        sampler = build_sampler(data_source = data_source, 
                                sampler_type = sampler_type, 
                                batch_size = batch_size, 
                                num_instances = cfg.DATALOADER.TRAIN.NUM_INSTANCES,
                                source_domains = source_domains,
                                need_day_night = need_day_night)
        data_loader = DataLoader(
            dataset = DatasetWrapper(cfg = cfg, data_source = data_source, transform = transform, is_train = is_train), 
            batch_size = batch_size, 
            sampler = sampler, 
            num_workers = cfg.DATALOADER.NUM_WORKERS, 
            collate_fn = train_collate_fn
        )
    else:
        data_loader = DataLoader(
            dataset = DatasetWrapper(cfg = cfg, data_source = data_source, transform = transform, is_train = is_train), 
            batch_size = batch_size, 
            num_workers = cfg.DATALOADER.NUM_WORKERS, 
            collate_fn = test_collate_fn,
            shuffle=False
        )
    
    assert len(data_loader) > 0, "Empty data loader"
    return data_loader


class DataManager:
    def __init__(self, cfg):
        self.datasets = build_dataset(cfg)    # creates a list of datasets
        self.source_domains = cfg.DATASET.SOURCE_DOMAINS if hasattr(cfg.DATASET, 'SOURCE_DOMAINS') else []
        self.target_domains = cfg.DATASET.TARGET_DOMAINS if hasattr(cfg.DATASET, 'TARGET_DOMAINS') else []

        self.train_data = []
        self.gallery_data = []
        self.query_data = []

        for dataset in self.datasets:
            self.train_data.extend(dataset.train_data)
            self.gallery_data.extend(dataset.gallery_data)
            self.query_data.extend(dataset.query_data)

        # train_global_ids = self.convert_to_global_id(self.train_data)
        test_global_ids = self.convert_to_global_id(self.query_data + self.gallery_data)

        transform_train = build_transform(cfg, is_train = True)
        transform_test = build_transform(cfg, is_train = False)
        
        self.data_loader_train = build_data_loader(
            cfg = cfg, 
            sampler_type = cfg.DATALOADER.TRAIN.SAMPLER, 
            data_source = self.train_data, 
            batch_size = cfg.DATALOADER.TRAIN.BATCH_SIZE, 
            transform = transform_train, 
            is_train = True,
            source_domains = self.get_source_domains
        )

        self.data_loader_test = build_data_loader(
            cfg = cfg, 
            data_source = self.query_data + self.gallery_data, 
            batch_size = cfg.DATALOADER.TEST.BATCH_SIZE, 
            transform = transform_test, 
            is_train = False,
        )
      
        self.len_query = len(self.query_data)

    @property
    def num_classes(self):
        num_classes_dict = {}
        for dataset in self.datasets:
            num_classes_dict[dataset.domain_label] = dataset.num_train_aids
        return num_classes_dict

    @property
    def get_num_train_images(self):
        return len(self.train_data)
    
    @property
    def get_source_domains(self):
        return self.source_domains

    @property
    def get_target_domains(self):
        return self.target_domains
    
    @property
    def get_num_train_camids(self):
        return self.dataset.num_train_cams
    
    @property
    def get_num_train_viewids(self):
        return self.dataset.num_train_views
    
    @property
    def get_num_gallery_images(self):
        return self.dataset.num_gallery_imgs
    
    @property
    def get_num_gallery_aids(self):
        return self.dataset.num_gallery_aids
    
    @property
    def get_num_gallery_camids(self):
        return self.dataset.num_gallery_cams
    
    @property
    def get_num_gallery_viewids(self):
        return self.dataset.num_gallery_views
    
    @property
    def get_num_query_images(self):
        return self.dataset.num_query_imgs
    
    @property
    def get_num_query_aids(self):
        return self.dataset.num_query_aids
    
    @property
    def get_num_query_camids(self):
        return self.dataset.num_query_cams
    
    @property
    def get_num_query_viewids(self):
        return self.dataset.num_query_views

    def convert_to_global_id(self, dataset):
        domain_to_aids = defaultdict(set)
        for d in dataset:
            # Each "d" is a Datum with fields aid (local id) and domain_label (int)
            domain_to_aids[d.domain_label].add(d.aid)

        domain_label_offsets = {}
        running_offset = 0
        for dom in sorted(domain_to_aids.keys()):
            domain_label_offsets[dom] = running_offset
            running_offset += len(domain_to_aids[dom])


        # Build globally unique class names using domain-specific offsets
        # Each global class id = local aid + offset(domain_label)
        global_ids = set()
        for d in dataset:
            global_id = domain_label_offsets[d.domain_label] + d.aid
            d.aid = global_id
            global_ids.add(global_id)
        return global_ids


class DatasetWrapper(Dataset):
    def __init__(self, cfg, data_source, transform = None, is_train = False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        datum = self.data_source[idx]

        output = {
            "img_path": datum.img_path,
            "aid": datum.aid, 
            "camid": datum.camid, 
            "viewid": datum.viewid, 
            "domain": datum.domain_label,
            "time": datum.is_day  # Store time of day information (1 for day, 0 for night)
        }

        img = Image.open(datum.img_path).convert("RGB")
        img = self.transform(img)

        output["img"] = img

        return output
