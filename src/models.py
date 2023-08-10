from matplotlib.style import context
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn, einsum
import wandb
from modules import query_grid_cat, visualize_result


PRIMES = [265443567,805459861]  #1,
        
class HashTable(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS=FLAGS
        self.codebook_size=2 ** self.FLAGS.band_width
        self.codebook = nn.ParameterList([])
        # self.LODS = [2**L for L in range(FLAGS.base_lod,FLAGS.base_lod + FLAGS.num_LOD)]
        b = np.exp((np.log(self.FLAGS.max_grid_res) - np.log(self.FLAGS.min_grid_res)) / (self.FLAGS.num_LOD-1))
        self.LODS = [int(1 + np.floor(self.FLAGS.min_grid_res*(b**l))) for l in range(self.FLAGS.num_LOD)]
        for LOD in self.LODS:
            num_pts=LOD**2
            fts = torch.zeros(min(self.codebook_size, num_pts), self.FLAGS.feat_dim) #+ self.feature_bias
            fts += torch.randn_like(fts) * self.FLAGS.feature_std
            feat = nn.Parameter(fts)
            feat = feat.cuda()
            self.codebook.append(feat)

    def get_features(self, positions, LOD):
        x1, y1, x2, y2, w1, w2, w3, w4, points = positions
        res = self.LODS[LOD]
        npts = res**2
        grid_res = self.codebook[LOD].shape[0]
        if npts > grid_res:
            id1 = ((x1 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int()) % (grid_res)
            id2 = ((x2 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int()) % (grid_res)
            id3 = ((x1 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int()) % (grid_res)
            id4 = ((x2 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int()) % (grid_res)

            idis = torch.cat([id1,id2,id3,id4],dim=0)
            x1s = torch.cat([x1,x2,x1,x2],dim=0)
            y1s = torch.cat([y1,y1,y2,y2],dim=0)

            feats = self.get_coll(LOD, idis.long())
            reshaped_feats = feats.view(4, -1 , feats.shape[1])
        else:
            x1s = torch.cat([x1,x2,x1,x2],dim=0)
            y1s = torch.cat([y1,y1,y2,y2],dim=0)
            feats = self.get(LOD, x1s.long(), y1s.long())
            reshaped_feats = feats.view(4, -1 , feats.shape[1])

        return torch.einsum('a,ab->ab', w1 , reshaped_feats[0]) + torch.einsum('a,ab->ab', w2 , reshaped_feats[1]) \
                    + torch.einsum('a,ab->ab', w3 , reshaped_feats[2]) + torch.einsum('a,ab->ab', w4 , reshaped_feats[3])
    
    def get(self, LOD, x, y):
        return self.codebook[LOD][x + y * self.LODS[LOD]]
    
    def get_coding(self, pos, image_size):
        # Get the features from the grid
        features = query_grid_cat(pos, self, image_size)
        return features
    
    def get_coll(self, LOD, idx):
        return self.codebook[LOD][idx]

class SimpleModel(pl.LightningModule):
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS=FLAGS
        self.criterion = nn.MSELoss()
        if self.FLAGS.multiscale_type == "cat":
            feat_dim = self.FLAGS.feat_dim * self.FLAGS.num_LOD
        elif self.FLAGS.multiscale_type == "sum":
            feat_dim = self.FLAGS.feat_dim
        self.activations = {"RELU": nn.ReLU()}
        self.simplemlp = nn.Sequential(
            nn.Linear(feat_dim, FLAGS.hidden_dim),
            self.activations[self.FLAGS.activation],
            nn.Linear(FLAGS.hidden_dim, FLAGS.n_channels),
        )
        self.result = torch.zeros(self.FLAGS.image_size, self.FLAGS.image_size, self.FLAGS.n_channels, device=torch.device("cuda"), requires_grad=False)
        if self.FLAGS.use_grid:
            if self.FLAGS.grid_type=='HASH':
                self.init_hash_structure()
            else:
                print("NOT SUPPORTED")

    def forward(self, x):
        x = self.simplemlp(x)
        return x
    
    def init_hash_structure(self):
        self.grid = HashTable(self.FLAGS)

    def input_mapping(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x) @ B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_psnr(self, pred, target):
        return 10 * torch.log10((1 ** 2) / torch.mean((pred - target) ** 2))

    def training_step(self, train_batch, batch_idx):
        pos = train_batch['xy'].squeeze().view(-1,2)
        image = train_batch['rgb'].squeeze().view(-1,self.FLAGS.n_channels)
        b,c = image.shape

        # Get features from the grid
        features = self.grid.get_coding(pos, self.FLAGS.image_size)#, self.error.view(self.FLAGS.image_size,self.FLAGS.image_size,-1)[pos[:,0].long(),pos[:,1].long()])
        
        # Forward pass
        x1 = self.forward(features)

        # Compute MSE loss
        loss = self.criterion(x1 , image)

        # Compute error for visualization
        self.psnr = self.get_psnr(x1 , image).item()

        #LOGGING
        self.log('Training/loss', loss)
        self.log('Training/LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        self.log('Training/PSNR', self.psnr, prog_bar=True)

        plt.clf()

        return loss
    
    def on_train_epoch_end(self) -> None:
        if(self.FLAGS.display) and self.trainer.current_epoch % 100 == 0:
            # Display the result
            visualize_result(self.result, self.FLAGS, self.trainer.current_epoch)

        return super().on_train_epoch_end()
    
    def configure_optimizers(self):
        grid_params = []
        other_params = []

        if self.FLAGS.freeze_nn:
            for param in self.simplemlp.parameters():
                param.requires_grad = False
        if self.FLAGS.use_grid:
            grid_params = list(self.grid.codebook.parameters())
        other_params = list(self.simplemlp.parameters())
        crit = list(self.criterion.parameters())

        params = []
        if grid_params:
            params.append({'params': grid_params, 'lr': self.FLAGS.learning_rate * self.FLAGS.grid_lr_factor})
        if crit:
            params.append({'params': crit, 'lr': self.FLAGS.learning_rate})
        if other_params:
            params.append({'params': other_params, 'lr': self.FLAGS.learning_rate, 'weight_decay': self.FLAGS.weight_decay})

        self.optimizer = torch.optim.Adam(
            params,
            eps=1e-15,
        )

        return {'optimizer': self.optimizer}

    