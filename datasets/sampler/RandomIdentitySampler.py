import copy
import numpy as np
import random

from collections import defaultdict
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        """
        Randomly sample N identities, then for each identity, 
        randomly sample K instances, therefore the batch size is N*K.

        Args:
            data_source (list): A list of Datum objects.
            batch_size (int): The number of samples in a batch.
            num_instances (int): The number of instances per identity in a batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_aids_per_batch = self.batch_size // self.num_instances
        self.index_dict = defaultdict(list)    # a dict with integer keys and list values (e.g., {aid:[indices]})
                                               # {327: [0, 6, 50, 118, 641, 1022], ...}
        for index, datum in enumerate(self.data_source):
            aid = datum.aid
            self.index_dict[aid].append(index)
        self.aids = list(self.index_dict.keys())    # a list of all IDs

        # Estimate number of samples in an epoch. 
        self.length = 0
        for aid in self.aids:
            idxs = self.index_dict[aid]    # a list of indices for a given ID
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        self.length = (self.length // self.batch_size) * self.batch_size    # ensure the length is a multiple of batch size

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)    # {aid: [[], [], ...]}

        for aid in self.aids:
            idxs = copy.deepcopy(self.index_dict[aid])    # a list of indices for a given ID
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size = self.num_instances, replace = True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[aid].append(batch_idxs)
                    batch_idxs = []

        avai_aids = copy.deepcopy(self.aids)    # a list of available IDs
        final_idxs = []

        while len(avai_aids) >= self.num_aids_per_batch:
            selected_aids = random.sample(avai_aids, self.num_aids_per_batch)
            for aid in selected_aids:
                batch_idxs = batch_idxs_dict[aid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[aid]) == 0:
                    avai_aids.remove(aid)

        return iter(final_idxs)

    def __len__(self):
        return self.length