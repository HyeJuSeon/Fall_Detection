import torch
import torch.utils.data as data_utl

import numpy as np
import pickle

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_pickle(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def make_dataset(root, split, mode, num_classes=2):
    path = 'gdrive/MyDrive/sw_capstone/'
    dataset = []
    if mode == 'flow':
        # path += 'urfd_flow_i3d'
        path += 'aihub_flow_i3d'
    elif mode == 'pose':
        # path += 'urfd_pose_i3d'
        path += 'aihub_pose_i3d'
    else:
        # path += 'urfd_rgb_i3d'
        path += 'aihub_rgb_i3d'
    if split == 'training':
        vids = load_pickle(f'{path}/vid_{mode}_train.pkl')
        labels = load_pickle(f'{path}/label_{mode}_train.pkl')
    else:
        vids = load_pickle(f'{path}/vid_{mode}_val.pkl')
        labels = load_pickle(f'{path}/label_{mode}_val.pkl')
    remove_idx = [i for i in range(len(vids)) if len(vids[i]) != 64]
    vids = [vid for i, vid in enumerate(vids) if i not in remove_idx]
    labels = [label for i, label in enumerate(labels) if i not in remove_idx]
    # print(np.asarray(vids, dtype=np.float32).shape, np.asarray(labels, dtype=np.float32).shape)
    return (np.asarray(vids, dtype=np.float32), np.asarray(labels, dtype=np.float32))

  
class Dataset(data_utl.Dataset):

    def __init__(self, root, split, mode, transforms=None):
      
        self.data = make_dataset(root=root, split=split, mode=mode)
        self.mode = mode
        self.root = root
        self.split = split
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        imgs, labels = self.data
        img, label = imgs[index], labels[index]
        # print(index, img.shape, label.shape)
        img = self.transforms(img)
        return video_to_tensor(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data[0])
