from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os

from .ops import to_cv, sort_keypoints

default_colors = {
    'g': '#4ade80',
    'r': '#ef4444',
    'b': '#3b82f6',
    'y': '#fbbf24',  # yellow
    'p': '#a855f7',  # purple
    'o': '#fb923c',  # orange
    'c': '#06b6d4',  # cyan
    'm': '#ec4899',  # magenta
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_color(color, n=None, cmap='hsv'):
    """Resolve a color specification into a matplotlib-compatible value.

    When *n* is None, returns a single resolved color string.
    When *n* is given, always returns a **list** of length *n*.
    """
    if isinstance(color, list):
        return [_resolve_color(c) for c in color]

    if isinstance(color, str):
        resolved = default_colors.get(color, color)
        if n is None:
            return resolved
        return [resolved] * n

    if color is None and n is not None:
        if n == 0:
            return []
        rainbow = plt.get_cmap(cmap)
        return [rainbow(i / n) for i in range(n)]

    # color is already a valid matplotlib color (tuple, ndarray, etc.)
    if n is None:
        return color
    return [color] * n


def _to_numpy(x):
    """Convert *x* to a numpy array. Passes through None and ndarrays."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _validate_keypoints(kpts):
    """Return a (N, 2+) numpy array, or None for empty / None inputs."""
    kpts = _to_numpy(kpts)
    if kpts is None or kpts.size == 0:
        return None
    if kpts.ndim == 3:
        kpts = kpts.squeeze(0)
    return kpts


def _get_fig_ax(fig, ax):
    """Return (fig, ax), falling back to the current figure and its axes."""
    if fig is not None and ax is not None:
        return fig, ax
    fig = plt.gcf()
    ax = fig.axes
    if not ax:
        raise RuntimeError(
            "No axes found on current figure. Call plot_pair or plot_image first."
        )
    return fig, ax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_keypoints(image, keypoints, color='g', size=5):
    if keypoints.shape[1] == 0:
        return image

    col = _resolve_color(color)
    imgs = []
    for B in range(keypoints.shape[0]):
        kps = keypoints[B].unsqueeze(0)
        im = image[B]
        im = torchvision.utils.draw_keypoints(im, kps, radius=size, colors=col)
        imgs.append(im)
    return torch.stack(imgs)


def plot_pair(img0, img1, figsize=(20, 10), fig=None, ax=None, title=None,
              vertical=False, gray=True, title_fontsize=None) -> tuple[plt.Figure, list[plt.Axes]]:
    if img0 is None or img1 is None:
        raise ValueError("img0 and img1 must not be None")

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

    for a in ax:
        a.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)

    if title is not None:
        kw = {}
        if title_fontsize is not None:
            kw['fontsize'] = title_fontsize
        fig.suptitle(title, **kw)
    return fig, ax


def plot_image(image, figsize=(10, 10), fig=None, ax=None, title=None,
               gray=True, title_fontsize=None) -> tuple[plt.Figure, list[plt.Axes]]:
    if image is None:
        raise ValueError("image must not be None")

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if gray:
        ax.imshow(to_cv(image, to_gray=gray), cmap='gray')
    else:
        ax.imshow(to_cv(image))

    ax.axis('off')

    if title is not None:
        kw = {}
        if title_fontsize is not None:
            kw['fontsize'] = title_fontsize
        fig.suptitle(title, **kw)
    return fig, ax


def plot_depth_pair(img0, img1, fig=None, ax=None, title=None,
                    cmap='jet', alpha=0.5) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, ax = _get_fig_ax(fig, ax)

    ax[0].imshow(to_cv(img0, to_gray=True), cmap=cmap, alpha=alpha)
    ax[1].imshow(to_cv(img1, to_gray=True), cmap=cmap, alpha=alpha)

    if title is not None:
        fig.suptitle(title)
    return fig, ax


def plot_depth(depth, fig=None, ax=None, title=None,
               cmap='jet', alpha=0.5) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, ax = _get_fig_ax(fig, ax)

    # ax may be a list (from plot_pair) or a single Axes (from plot_image)
    target = ax[0] if isinstance(ax, (list, np.ndarray)) else ax
    target.imshow(depth, cmap=cmap, alpha=alpha)

    if title is not None:
        fig.suptitle(title)
    return fig, ax


def plot_keypoints(keypoints0=None, keypoints1=None, fig=None, ax=None,
                   color=None, kps_size=5, all_colors=None,
                   marker=None, alpha=1.0, cmap='hsv',
                   **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, ax = _get_fig_ax(fig, ax)

    # Convert to numpy BEFORE sorting (sort_keypoints uses np.lexsort)
    keypoints0 = _validate_keypoints(keypoints0)
    keypoints1 = _validate_keypoints(keypoints1)

    if keypoints0 is None and keypoints1 is None:
        return fig, ax

    if keypoints0 is not None and keypoints1 is not None:
        keypoints0, keypoints1 = sort_keypoints(keypoints0, keypoints1)

    scatter_kw = dict(s=kps_size, alpha=alpha, **kwargs)
    if marker is not None:
        scatter_kw['marker'] = marker

    if keypoints0 is not None:
        colors0 = all_colors if all_colors is not None else _resolve_color(color, n=len(keypoints0), cmap=cmap)
        ax[0].scatter(keypoints0[:, 0], keypoints0[:, 1], c=colors0, **scatter_kw)

    if keypoints1 is not None:
        colors1 = all_colors if all_colors is not None else _resolve_color(color, n=len(keypoints1), cmap=cmap)
        ax[1].scatter(keypoints1[:, 0], keypoints1[:, 1], c=colors1, **scatter_kw)

    return fig, ax


def plot_matches(mkpts0, mkpts1, fig=None, ax=None, color=None,
                 linewidth=0.5, alpha=None, cmap='hsv', **kwargs):
    fig, ax = _get_fig_ax(fig, ax)

    mkpts0 = _to_numpy(mkpts0)
    mkpts1 = _to_numpy(mkpts1)

    if mkpts0 is None or mkpts1 is None or len(mkpts0) == 0 or len(mkpts1) == 0:
        return fig, ax

    if len(mkpts0) != len(mkpts1):
        raise ValueError(
            f"mkpts0 and mkpts1 must have the same length, got {len(mkpts0)} and {len(mkpts1)}"
        )

    colors = _resolve_color(color, n=len(mkpts0), cmap=cmap)

    mkpts0, mkpts1 = sort_keypoints(mkpts0, mkpts1)

    con_kw = dict(linewidth=linewidth, **kwargs)
    if alpha is not None:
        con_kw['alpha'] = alpha

    for i, (mkp0, mkp1) in enumerate(zip(mkpts0, mkpts1)):
        con = ConnectionPatch(
            xyA=mkp0, xyB=mkp1,
            coordsA="data", coordsB="data",
            axesA=ax[0], axesB=ax[1],
            color=colors[i], **con_kw)
        con.set_in_layout(False)
        ax[0].add_artist(con)
    ax[1].set_zorder(-1)

    return fig, ax


def add_text(text, fig=None, ax=None, fontsize=12, color='black',
             bg_color='white', bg_alpha=0.5, x=0, y=0,
             ha='left', va='top', **kwargs):
    fig, ax = _get_fig_ax(fig, ax)
    target = ax[0] if isinstance(ax, (list, np.ndarray)) else ax
    target.text(
        x, y, str(text),
        color=color, fontsize=fontsize, ha=ha, va=va,
        bbox=dict(facecolor=bg_color, alpha=bg_alpha, edgecolor=bg_color),
        **kwargs,
    )
    return fig, ax


def show():
    if plt.get_fignums():
        plt.tight_layout()
    plt.show()


def save(path, dpi=None, bbox_inches=None, close=True):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    save_kw = {}
    if dpi is not None:
        save_kw['dpi'] = dpi
    if bbox_inches is not None:
        save_kw['bbox_inches'] = bbox_inches

    plt.tight_layout()
    plt.savefig(path, **save_kw)
    if close:
        plt.close()
