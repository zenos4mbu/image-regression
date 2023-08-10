import torch
from torch import einsum, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math
from scipy.interpolate import RegularGridInterpolator
import scipy
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
import wandb
from PIL import Image
import io
import pdb

PRIMES = [265443567,805459861]
# BOX_OFFSETS_2D = torch.tensor([[i,j] for i in [0, 1] for j in [0, 1]],device='cuda')

def query_grid_sum(points, grid, image_size):
    feats = []
    #Iterate in every level of detail resolution
    for LOD, res in enumerate(grid.LODS):
        # Convert the points to the grid's coordinate system
        positions = bilinear_interpolation(res, points, image_size)

        # Get the features for each point
        features = grid.get_features(positions, LOD)
        feats.append((torch.unsqueeze(features, dim =-1)))
    ficiur = torch.cat(feats, -1)
    return ficiur.sum(-1)

def query_grid_cat(points, grid, image_size):
    feats = []
    #Iterate in every level of detail resolution
    for LOD, res in enumerate(grid.LODS):
        # Convert the points to the grid's coordinate system
        positions = bilinear_interpolation(res, points, image_size)

        # Get the features for each point
        features = grid.get_features(positions, LOD)
        feats.append(features)
    ficiur = torch.cat(feats, -1)
    return ficiur


def bilinear_interpolation(res, points, image_size):
    x1, y1, x2, y2, points = get_vertexes(res, points, image_size)
    # Compute the weights for each of the four points
    w1 = (x2 - points[:, 0]) * (y2 - points[:, 1])
    w2 = (points[:, 0] - x1) * (y2 - points[:, 1])
    w3 = (x2 - points[:, 0]) * (points[:, 1] - y1)
    w4 = (points[:, 0] - x1) * (points[:, 1] - y1)

    return x1, y1, x2, y2, w1, w2, w3, w4, points

def get_vertexes(res, original_points, image_size):
    # Convert the points to the grid's coordinate system
    points = original_points * ((res - 1 - 1e-5) / (image_size - 1 - 1e-5))

    # Get the four surrounding points for each point
    x1 = torch.floor(points[:, 0]).int()
    y1 = torch.floor(points[:, 1]).int()
    x2 = x1 + 1
    y2 = y1 + 1

    return x1, y1, x2, y2, points


def visualize_result(image, FLAGS, current_epoch):
    plt.clf()
    # Create a figure
    fig = plt.figure()

    # Create the first subplot for imshow
    plt.xlim(0, FLAGS.image_size)
    plt.ylim(0, FLAGS.image_size)
    plt.imshow(image.clip(0, 1).permute(1, 0, 2).detach().cpu().numpy())
    plt.set_aspect('equal')  # Set aspect ratio to equal
    plt.title("Reconstruction")
    plt.axis('off')

    # SISTEMATE QUI

    # output_folder = "EXP_nuovi/{}".format(FLAGS.exp_name)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # plt.savefig("EXP_nuovi/{}/centroids{:05d}.png".format(FLAGS.exp_name,current_epoch))

    # Convert the plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    # Log the image to wandb
    wandb.log({"centroids_plot": wandb.Image(img, caption="Centroids Plot")})

    buf.close()
    plt.close()

def visualize_results(pred, true):
    pred_true = np.hstack((pred.detach().cpu().numpy(), true.detach().cpu().numpy()))
    pred_true = cv2.cvtColor(pred_true, cv2.COLOR_RGB2BGR)
    cv2.imshow("pred true", pred_true)
