import h5py
import numpy as np
import torch

from torch.utils.data import Dataset,DataLoader

# Path of the pre-extracted GoogLeNet pool5 features
def h5_path(dataset):
    return f'./data/eccv16_dataset_{dataset.lower()}_google_pool5.h5'

# Load split
def load_split(dataset, split_dir='./splits'):
    outputs = []
    with open(f'{split_dir}/{dataset}_splits.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            _,_,train_videos,test_videos = line.strip().split('/')
            train_videos = train_videos.split(',')
            test_videos = test_videos.split(',')
            outputs.append((train_videos,test_videos))
    return outputs

# Hold out part of the training videos for checkpoint selection (deterministic per split)
def split_validation(train_videos, val_ratio, seed):
    n_val = max(1, int(round(len(train_videos) * val_ratio)))
    rng = np.random.RandomState(seed)
    val_idx = set(rng.choice(len(train_videos), size=n_val, replace=False).tolist())
    train_part = [v for i,v in enumerate(train_videos) if i not in val_idx]
    val_part = [v for i,v in enumerate(train_videos) if i in val_idx]
    return train_part, val_part

# Create input,ground truth pair
def load_h5(videos,data_path,dataset_name):
    features = []
    gtscores = []
    dataset_names = []

    with h5py.File(data_path,'r') as hdf:
        for video in videos:
            feature = hdf[video]['features'][()]
            gtscore = hdf[video]['gtscore'][()]

            features.append(feature)
            gtscores.append(gtscore)
            dataset_names.append(dataset_name)
    return features,gtscores,dataset_names

# Create Dataset
class VSdataset(Dataset):
    def __init__(self,data,video_nums,transform=None):
        features,gtscores,dataset_names = data
        self.features = features
        self.gtscores = gtscores
        self.dataset_names = dataset_names
        self.video_nums = video_nums
        self.transform = transform
    def __len__(self):
        return len(self.video_nums)
    def __getitem__(self,idx):
        output_feature = torch.from_numpy(self.features[idx]).float()
        output_feature = output_feature.unsqueeze(0).expand(3,-1,-1)
        if self.transform is not None:
            output_feature=  self.transform(output_feature)
        return torch.unsqueeze(output_feature,0),torch.from_numpy(self.gtscores[idx]).float(),self.dataset_names[idx],self.video_nums[idx]

def collate_fn(sample):
    return sample[0]

def make_loader(videos, data_path, dataset, shuffle):
    data = load_h5(videos=videos,data_path=data_path,dataset_name=dataset)
    return DataLoader(VSdataset(data=data,video_nums=videos),batch_size=1,shuffle=shuffle,collate_fn=collate_fn)

# Create Dataloaders: one (train, val, test) triple per split.
# With val_ratio=0 the validation loader is None (used by evaluation-only scripts).
def create_dataloader(dataset, split_dir='./splits', val_ratio=0.125, seed=123456):
    loaders = []
    data_path = h5_path(dataset)

    for split_id,(train_videos,test_videos) in enumerate(load_split(dataset=dataset, split_dir=split_dir)):
        val_loader = None
        if val_ratio > 0:
            train_videos,val_videos = split_validation(train_videos, val_ratio, seed + split_id)
            assert not set(val_videos) & set(test_videos)
            val_loader = make_loader(val_videos, data_path, dataset, shuffle=False)

        train_loader = make_loader(train_videos, data_path, dataset, shuffle=True)
        test_loader = make_loader(test_videos, data_path, dataset, shuffle=False)
        loaders.append((train_loader,val_loader,test_loader))
    return loaders
