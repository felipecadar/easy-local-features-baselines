from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision

from .ops import to_cv

default_colors = {
    'g': '#4ade80',
    'r': '#ef4444',
    'b': '#3b82f6',
}

def draw_keypoints(image, keypoints, color='g', size=5):
    print(keypoints.shape)
    # draw keypoints on the image
    imgs = []
    for B in range(keypoints.shape[0]):
        kps = keypoints[B].unsqueeze(0)
        im = image[B]
        
        im = torchvision.utils.draw_keypoints(im, kps, radius=size, colors=default_colors[color])
        
        imgs.append(im)
    return torch.stack(imgs)
        

def plot_pair(img0, img1, figsize=(20, 10), fig=None, ax=None, title=None, vertical=False, gray=True) -> tuple[plt.Figure, list[plt.Axes]]:
    if fig is None:
        if vertical:
            fig, ax = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            
    if gray:
        ax[0].imshow(to_cv(img0, to_gray=gray), cmap='gray')
        ax[1].imshow(to_cv(img1, to_gray=gray), cmap='gray')
    else:
        ax[0].imshow(to_cv(img0))
        ax[1].imshow(to_cv(img1))
    
    # remove border
    for a in ax:
        a.axis('off')
    # remove space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plot_depth(img0, img1, fig=None, ax=None, title=None) -> tuple[plt.Figure, list[plt.Axes]]:
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes

    ax[0].imshow(to_cv(img0, to_gray=True), cmap='jet', alpha=0.5)
    ax[1].imshow(to_cv(img1, to_gray=True), cmap='jet', alpha=0.5)
    
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plot_keypoints(keypoints0=None, keypoints1=None, fig=None, ax=None, color=None, kps_size=5, all_colors=None, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
    rainbow = plt.get_cmap('hsv')
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes
    
    if keypoints0 is not None:        
        if all_colors is None:
            if isinstance(color, str) and color in default_colors:
                all_colors0 = [default_colors[color] for _ in range(len(keypoints0))]
            else:
                all_colors0 = [rainbow(i / len(keypoints0)) for i in range(len(keypoints0))]
        else:
            all_colors0 = all_colors

        if isinstance(keypoints0, torch.Tensor):
            keypoints0 = keypoints0.detach().cpu().numpy()
        if len(keypoints0.shape) == 3:
            keypoints0 = keypoints0.squeeze(0)
        ax[0].scatter(keypoints0[:, 0], keypoints0[:, 1], s=kps_size, c=all_colors0, **kwargs)

    if keypoints1 is not None:
        if all_colors is None:
            if isinstance(color, str) and color in default_colors:
                all_colors1 = [default_colors[color] for _ in range(len(keypoints1))]
            else:
                all_colors1 = [rainbow(i / len(keypoints1)) for i in range(len(keypoints1))]
        else:
            all_colors1 = all_colors
            
        if isinstance(keypoints1, torch.Tensor):
            keypoints1 = keypoints1.detach().cpu().numpy()
        if len(keypoints1.shape) == 3:
            keypoints1 = keypoints1.squeeze(0)
        ax[1].scatter(keypoints1[:, 0], keypoints1[:, 1], s=kps_size, c=all_colors1, **kwargs)

    return fig, ax

def plot_matches(mkpts0, mkpts1, fig=None, ax=None, color=None, **kwargs):
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes
    
    if (isinstance(color, str)) and (color in default_colors):
        color = default_colors[color]
    
    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.detach().cpu().numpy()
    if isinstance(mkpts1, torch.Tensor):
        mkpts1 = mkpts1.detach().cpu().numpy()
        
    if color is None:
        rainbow = plt.get_cmap('hsv')
        color = [rainbow(i / len(mkpts0)) for i in range(len(mkpts0))]
    
    if not isinstance(color, list):
        color = [color for _ in range(len(mkpts0))]
        
    # if we have the color in the default colors, replace it
    for i, c in enumerate(color):
        if c in default_colors:
            color[i] = default_colors[c]
    
    for i, (mkp0, mkp1) in enumerate(zip(mkpts0, mkpts1)):
        con = ConnectionPatch(
            xyA=mkp0, xyB=mkp1, 
            coordsA="data", coordsB="data",
            axesA=ax[0], axesB=ax[1], 
            color=color[i], linewidth=1.5, **kwargs)
        con.set_in_layout(False) # remove from layout calculations
        ax[0].add_artist(con)
    ax[1].set_zorder(-1)
    
    return fig, ax

def show():
    plt.tight_layout()
    plt.show()
    
def save(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()