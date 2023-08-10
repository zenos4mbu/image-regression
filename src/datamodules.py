import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor

import torch,imageio,cv2
from torch.utils.data import Dataset
import random


def create_grid(h, w, device="cpu"):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


class ImageDataset(Dataset):

    def __init__(self, FLAGS, image_path, img_dim, trainset_size, batch_size):
        self.FLAGS = FLAGS
        self.trainset_size = trainset_size
        self.batch_size = batch_size
        self.img_dim = img_dim
        image = load(image_path).astype(np.float32)
        if len(image.shape)==2:
            image = image[:,:, np.newaxis]
        h, w, c = image.shape
        img = cv2.resize(image, (img_dim, img_dim))
        self.img = img

    def __getitem__(self, idx):
        image = self.img# / 255
        y, x = torch.meshgrid(torch.arange(0, self.img_dim), torch.arange(0, self.img_dim), indexing='ij')
        grid = torch.stack((x, y), -1).float()  # +0.5
        # idxs = torch.randperm(grid[:,:,0].nelement())
        image = torch.tensor(image, dtype=torch.float32)
        grid = grid.view(-1, 2).view(self.img_dim, self.img_dim, 2)
        # grid = grid.view(-1, 2)[idxs].view(self.img_dim, self.img_dim, 2)
        image= image.unsqueeze(-1)
        # image = image.view(-1, self.FLAGS.n_channels)[idxs].view(self.img_dim, self.img_dim, self.FLAGS.n_channels)
        image = image.view(-1, self.FLAGS.n_channels).view(self.img_dim, self.img_dim, self.FLAGS.n_channels)
        sample = {'rgb': image,
                      'xy': grid}
        # return grid, torch.tensor(image, dtype=torch.float32)#, batches
        return sample

    def __len__(self):
        return self.trainset_size
    
    _img_suffix = ['png','jpg','jpeg','bmp','tif']

    def load(path):
        suffix = path.split('.')[-1]
        if suffix in _img_suffix:
            img =  np.array(Image.open(path))#.convert('L')
            scale = 256.**(1+np.log2(np.max(img))//8)-1
            return img/scale
        elif 'exr' == suffix:
            return imageio.imread(path)
        elif 'npy' == suffix:
            return np.load(path)

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, FLAGS):#, data_dir: str = "./input/frattaglia.png"):
        super().__init__()
        self.FLAGS = FLAGS
        self.data_dir = FLAGS.image_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage = None):
        # self.dataset = ImageDataset2(self.FLAGS, self.FLAGS.batch_size, HW=self.FLAGS.image_size)
        self.dataset = ImageDataset(self.FLAGS, self.data_dir, self.FLAGS.image_size, self.FLAGS.trainset_size, self.FLAGS.batch_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size, shuffle=True)
    
_img_suffix = ['png','jpg','jpeg','bmp','tif']

def load(path):
    suffix = path.split('.')[-1]
    if suffix in _img_suffix:
        img =  np.array(Image.open(path))#.convert('L')
        scale = 256.**(1+np.log2(np.max(img))//8)-1
        return img/scale
    elif 'exr' == suffix:
        return imageio.imread(path)
    elif 'npy' == suffix:
        return np.load(path)

    
def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

class ImageDataset2(Dataset):
    def __init__(self, cfg, batchsize, split='train', continue_sampling=False, tolinear=True, HW=-1, perscent=1.0, delete_region=None, mask=None):
        datadir = cfg.image_path
        self.batchsize = batchsize
        self.continue_sampling = continue_sampling
        img = load(datadir).astype(np.float32)
        if HW > 0:
            img = cv2.resize(img, (HW, HW))

        if tolinear:
            img = srgb_to_linear(img)
            
        # self.importance_map = self.compute_importance_map(img)  # Compute the importance map after resizing and converting to linear

        self.img = torch.from_numpy(img)

        H, W = self.img.shape[:2]

        y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        self.coordiante = torch.stack((x, y), -1).float()  # +0.5

        n_channel = self.img.shape[-1]
        self.image = self.img
        self.img, self.coordiante = self.img.reshape(H * W, -1), self.coordiante.reshape(H * W, 2)
        
        # if continue_sampling:
        #     coordiante_tmp = self.coordiante.view(1,1,-1,2)/torch.tensor([W,H])*2-1.0
        #     self.img = F.grid_sample(self.img.view(1,H,W,-1).permute(0,3,1,2),coordiante_tmp, mode='bilinear', align_corners=True).reshape(self.img.shape[-1],-1).t()
            
            
        if 'train'==split:
            self.mask = torch.ones_like(y)>0
            if mask is not None:
                self.mask = mask>0
                print(torch.sum(mask)/1.0/HW/HW)
            elif delete_region is not None:
                
                if isinstance(delete_region[0], list):
                    for item in delete_region:
                        t_l_x,t_l_y,width,height = item
                        self.mask[t_l_y:t_l_y+height,t_l_x:t_l_x+width] = False
                else:
                    t_l_x,t_l_y,width,height = delete_region
                    self.mask[t_l_y:t_l_y+height,t_l_x:t_l_x+width] = False
            else:
                index = torch.randperm(len(self.img))[:int(len(self.img)*perscent)] 
                self.mask[:] = False
                self.mask.view(-1)[index] = True
            self.mask = self.mask.view(-1)
            self.image, self.coordiante = self.img[self.mask], self.coordiante[self.mask]
        else:
            self.image = self.img
            

        self.HW = [H,W]

        self.scene_bbox = [[0., 0.], [W, H]]
        # cfg.aabb = self.scene_bbox
        #

    def __len__(self):
        return 10000
    
    def compute_importance_map(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
        grad_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        importance_map = grad_magnitude / np.sum(grad_magnitude)
        return importance_map
    

    def __getitem__(self, idx):
        H,W = self.HW 
        device = self.image.device
        # idx = np.random.choice(np.arange(H * W), size=self.batchsize, p=self.importance_map.flatten())
        idx = torch.randint(0,len(self.image),(self.batchsize,), device=device)

        if self.continue_sampling:
            coordinate = self.coordiante[idx] +  torch.rand((self.batchsize,2))#-0.5
            coordinate_tmp = (coordinate.view(1,1,self.batchsize,2))/torch.tensor([W,H],device=device)*2-1.0
            rgb = F.grid_sample(self.img.view(1,H,W,-1).permute(0,3,1,2),coordinate_tmp, mode='bilinear', 
                                align_corners=False, padding_mode='border').reshape(self.img.shape[-1],-1).t()
            sample = {'rgb': rgb,
                      'xy': coordinate}
        else:
            sample = {'rgb': self.image[idx],
                      'xy': self.coordiante[idx]}

        return sample