import copy
import numpy as np
import random

from collections import defaultdict
from torch.utils.data.sampler import Sampler

class DayNightRandomIdentitySampler(Sampler):
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
        self.index_dict_day = defaultdict(list)    # a dict with integer keys and list values (e.g., {aid:[indices]})
                                               # {327: [0, 6, 50, 118, 641, 1022], ...}
        self.index_dict_night = defaultdict(list)

        for index, datum in enumerate(self.data_source):
            aid = datum.aid
            if datum.is_day:
                self.index_dict_day[aid].append(index)
            else:
                self.index_dict_night[aid].append(index)
        self.aids = list(set(list(self.index_dict_day.keys()) + list(self.index_dict_night.keys())))   # a list of all IDs

        # Estimate number of samples in an epoch. 
        self.length = 0
        for aid in self.aids:
            idxs = self.index_dict_day[aid] + self.index_dict_night[aid]    # a list of indices for a given ID
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        self.length = (self.length // self.batch_size) * self.batch_size    # ensure the length is a multiple of batch size

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)    # {aid: [[], [], ...]}

        for aid in self.aids:
            day_idxs = copy.deepcopy(self.index_dict_day.get(aid, []))    # a list of indices for a given ID
            night_idxs = copy.deepcopy(self.index_dict_night.get(aid, []))

            if len(day_idxs) == 0 and len(night_idxs) == 0:
                continue

            if len(day_idxs) == 0 or len(night_idxs) == 0:
                day_night_num = self.num_instances
            else:
                # num_instance is assumed to be even
                day_night_num = self.num_instances // 2
            if len(day_idxs) < day_night_num and len(day_idxs) > 0:
                day_idxs = np.random.choice(day_idxs, size = day_night_num, replace = True)
            if len(night_idxs) < day_night_num and len(night_idxs) > 0:
                night_idxs = np.random.choice(night_idxs, size = day_night_num, replace = True)
            random.shuffle(day_idxs)
            random.shuffle(night_idxs)
            batch_idxs = []
            larger_idxs = day_idxs if len(day_idxs) >= len(night_idxs) else night_idxs
            smaller_idxs = night_idxs if len(day_idxs) >= len(night_idxs) else day_idxs
            for i, idx in enumerate(larger_idxs):
                batch_idxs.append(idx)
                if i < len(smaller_idxs):
                    batch_idxs.append(smaller_idxs[i])
                if day_night_num == self.num_instances:
                    if len(batch_idxs) == day_night_num:
                        batch_idxs_dict[aid].append(batch_idxs)
                        batch_idxs = []
                else:
                    if len(batch_idxs) == day_night_num * 2:
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