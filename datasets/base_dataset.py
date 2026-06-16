import gdown
import os
import zipfile
from collections import Counter
from tabulate import tabulate
from utils.color_check import check_color_diff
from utils.tools import listdir_nonhidden

def get_dataset_info(dataset):
    aids_list, camids_list, viewids_list = [], [], []
    num_day = 0
    num_night = 0

    aid_count = {}

    for idx, entry in enumerate(dataset):
        aids_list.append(entry.aid)
        camids_list.append(entry.camid)
        viewids_list.append(entry.viewid)
        aid_count[entry.aid] = aid_count.get(entry.aid, 0) + 1
        if entry.is_day == 1:
            num_day += 1
        elif entry.is_day == 0:
            num_night += 1


    aids = set(aids_list)
    camids = set(camids_list)
    viewids = set(viewids_list)
    
    num_of_images = len(dataset)
    num_of_aids = len(aids) # num of classes for animal ReID
    num_of_camids = len(camids)
    num_of_viewids = len(viewids)
    return num_of_images, num_of_aids, num_of_camids, aid_count, num_day, num_night


class Datum:
    def __init__(self, img_path, aid, camid, viewid, domain_label, need_day_night=False):
        """
        Data instance for Animal ReID which defines the basic attributes.

        Args:
            img_path (str): Image path.
            aid (int): Animal ID.
            camid (str): Camera ID.
            viewid (int): View ID.
            domain_label (int): Domain label.
        """
        assert isinstance(img_path, str)
        assert os.path.isfile(img_path)
        assert isinstance(aid, int)
        assert isinstance(camid, str)
        assert isinstance(viewid, int)

        self._img_path = img_path
        self._aid = aid
        self._camid = camid
        self._viewid = viewid
        self._domain_label = domain_label
        if need_day_night:
            self.is_day = 1 if check_color_diff(self._img_path) else 0
        else:
            self.is_day = -1  # Unknown

    @property
    def img_path(self):
        return self._img_path
    
    @property
    def aid(self):
        return self._aid
    
    @aid.setter
    def aid(self, value):
        self._aid = value
    
    @property
    def camid(self):
        return self._camid
    
    @property
    def viewid(self):
        return self._viewid
    
    @property
    def domain_label(self):
        return self._domain_label

    @property
    def is_day_ground(self):
        # Ground truth for day/night based on folder name
        if "day" in self._img_path.lower():
            return 1
        elif "night" in self._img_path.lower():
            return 0
        else:
            return -1  # Unknown

class DatasetBase:
    def __init__(
        self, 
        dataset_dir, 
        domain,
        data_url=None, 
        train_data=None, 
        gallery_data=None, 
        query_data=None, 
    ):
        """
        Base class of Animal ReID dataset.

        Args:
            dataset_dir (str): Dataset directory.
            data_url (str): Dataset URL.
            train_data (list): A list of attributes of the training dataset.
            gallery_data (list): A list of attributes of the gallery dataset.
            query_data (list): A list of attributes of the query dataset.
            domain (string): Domain label of this dataset, same as the name of the Dataset.
            num_classes (int): Number of aniaml IDs in the training set.
            class_names (list): List of class names of the training set.
        """
        self._dataset_dir = dataset_dir
        self._data_url = data_url
        self._train_data = train_data
        self._gallery_data = gallery_data
        self._query_data = query_data
        self._domain = domain
    @property
    def dataset_dir(self):
        return self._dataset_dir
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def data_url(self):
        return self._data_url
    
    @property
    def train_data(self):
        return self._train_data
    
    @property
    def gallery_data(self):
        return self._gallery_data
    
    @property
    def query_data(self):
        return self._query_data
    
    def download_data_from_gdrive(self, dst):
        gdown.download(self._data_url, dst, quiet=False)

        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(os.path.dirname(dst))
        zip_ref.close()
        print("File Extracted to {}".format(os.path.dirname(dst)))
        os.remove(dst)

    def check_input_domains(self, source_domains, target_domain):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domain)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self._domains:
                raise ValueError(
                    "Input Domain Must Belong to {}, " "but Got [{}]".format(
                        self._domains, domain
                    )
                )
    def show_dataset_info(self, need_day_night=False):
        train_info = get_dataset_info(self._train_data)
        gallery_info = get_dataset_info(self._gallery_data)
        query_info = get_dataset_info(self._query_data)

        if not need_day_night:
            headers = ["Subset", "# images", "# ids", "# cameras"]
            table = [
            ["train"] + list(train_info)[0:3], 
            ["gallery"] + list(gallery_info)[0:3], 
            ["query"] + list(query_info)[0:3]
            ]
        else:
            headers = ["Subset", "# images", "# ids", "# cameras", "# day", "# night"]
            table = [
                ["train"] + list(train_info)[0:3] + list(train_info)[-2:], 
                ["gallery"] + list(gallery_info)[0:3] + list(gallery_info)[-2:], 
                ["query"] + list(query_info)[0:3] + list(query_info)[-2:]
                ]
        print("Dataset statistics:")
        print(tabulate(table, headers = headers, tablefmt = "github"))

class BaseImageDataset(DatasetBase):
    """
    Base class for image-based ReID datasets.
    Absorbs the duplicated logic for loading images from train/gallery/query directories.
    """
    def __init__(self, cfg, dataset_dir, domain, domain_label, train_dir=None, gallery_dir=None, query_dir=None, need_day_night=False, relabel_train=True, verbose=True):
        self.cfg = cfg
        self._dataset_path = dataset_dir
        self._domain = domain
        self.domain_label = domain_label
        
        self.train_dir = train_dir
        self.gallery_dir = gallery_dir
        self.query_dir = query_dir
        self.need_day_night = need_day_night
        
        self._check_before_run()
        train_data = self.read_data(data_dir=self.train_dir, relabel=relabel_train)
        gallery_data = self.read_data(data_dir=self.gallery_dir, relabel=False)
        query_data = self.read_data(data_dir=self.query_dir, relabel=False)
        
        super().__init__(
            dataset_dir=self._dataset_path,
            train_data=train_data,
            gallery_data=gallery_data,
            query_data=query_data,
            domain=self._domain
        )
        
        if verbose:
            print(f"=> {self._domain} loaded")
            self.show_dataset_info(self.need_day_night)
            
        self.num_train_imgs, self.num_train_aids, self.num_train_cams, self.num_train_views, _, _ = get_dataset_info(self.train_data)
        self.num_gallery_imgs, self.num_gallery_aids, self.num_gallery_cams, self.num_gallery_views, _, _ = get_dataset_info(self.gallery_data)
        self.num_query_imgs, self.num_query_aids, self.num_query_cams, self.num_query_views, _, _ = get_dataset_info(self.query_data)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self._dataset_path):
            raise RuntimeError("'{}' is not available".format(self._dataset_path))

        if self.train_dir and not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if self.gallery_dir and not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if self.query_dir and not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def read_data(self, data_dir, relabel=False):
        def _load_data_from_directory(dir_path):
            files_list = listdir_nonhidden(path=dir_path)
            image_paths = []
            for file in files_list:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_p = os.path.join(dir_path, file)
                    image_paths.append(img_p)
            return image_paths
        
        data_paths = _load_data_from_directory(data_dir)
        label = 0
        aid2label = dict()
        aid_container, camid_container = set(), set()
        img_datums = []
        
        for img_p in data_paths:
            image_name, ext = os.path.splitext(os.path.basename(img_p))
            components_list = image_name.split("_")
            aid, camid = int(components_list[0]), components_list[1]
            aid_container.add(aid)
            camid_container.add(camid)
            if aid not in aid2label:
                aid2label[aid] = label
                label += 1
            
            if relabel:
                aid = aid2label[aid]
            
            img_datum = Datum(
                img_path=img_p, 
                aid=aid, 
                camid=camid, 
                viewid=-1,
                domain_label=self.domain_label,
                need_day_night=self.need_day_night
            )
            img_datums.append(img_datum)

        return img_datums
