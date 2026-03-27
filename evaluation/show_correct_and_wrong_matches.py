#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from typing import Tuple, List, Optional


def add_frame(img_in: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Adds a colored frame around an input image.

    Args:
        img_in (np.ndarray): A three-dimensional array representing the input image (height, width, channels).
        color (Tuple[int, int, int]): A tuple of three integers representing the RGB color of the frame.

    Returns:
        np.ndarray: A three-dimensional array representing the image with the added frame.
    """
    img = img_in.copy()

    w = int(np.round(0.01*img.shape[1]))

    # pad left-right
    pad_lr = np.tile(np.uint8(color).reshape(1, 1, 3), (img.shape[0], w, 1))
    img = np.concatenate([pad_lr, img, pad_lr], axis=1)

    # pad top-bottom
    pad_tb = np.tile(np.uint8(color).reshape(1, 1, 3), (w, img.shape[1], 1))
    img = np.concatenate([pad_tb, img, pad_tb], axis=0)

    return img


def show(
    db_imgs: List[np.ndarray],
    q_imgs: List[np.ndarray],
    TP: np.ndarray,
    FP: np.ndarray,
    M: Optional[np.ndarray] = None,
    n_correct: int = 1,
    n_wrong: int = 1,
    save_path: Optional[str] = None,
) -> None:
    """
    Displays a visual comparison of true positive and false positive image pairs
    from a database and query set. Optionally, a similarity matrix can be included.

    Args:
        db_imgs (List[np.ndarray]): A list of 3D arrays representing the database images (height, width, channels).
        q_imgs (List[np.ndarray]): A list of 3D arrays representing the query images (height, width, channels).
        TP (np.ndarray): A two-dimensional array containing the indices of true positive pairs.
        FP (np.ndarray): A two-dimensional array containing the indices of false positive pairs.
        M (Optional[np.ndarray], optional): A two-dimensional array representing the similarity matrix. Defaults to None.
        n_correct (int, optional): Number of correct matches to display. Defaults to 1.
        n_wrong (int, optional): Number of wrong matches to display. Defaults to 1.
        save_path (Optional[str], optional): Path to save the figure. If None, figure is displayed but not saved. Defaults to None.

    Returns:
        None: This function displays the comparison result using matplotlib.pyplot but does not return any value.
    """
    # true positive TP
    if(len(TP) == 0):
        print('No true positives found.')
        return

    print(f'\n  Creating visualization with {n_correct} correct and {n_wrong} wrong matches...')

    # Limit to available TPs
    n_correct = min(n_correct, len(TP))
    idx_tp = np.random.permutation(len(TP))[:n_correct]

    print(f'  Selected {n_correct} correct match(es) from {len(TP)} available')

    tp_pairs = []
    for i, idx in enumerate(idx_tp):
        db_idx = int(TP[idx, 0])
        q_idx = int(TP[idx, 1])
        print(f'    ✓ Correct match {i+1}: DB image {db_idx} <-> Query image {q_idx}')

        db_tp = db_imgs[db_idx]
        q_tp = q_imgs[q_idx]

        if db_tp.shape != q_tp.shape:
            q_tp = resize(q_tp.copy(), db_tp.shape, anti_aliasing=True)
            q_tp = np.uint8(q_tp*255)

        pair_img = add_frame(np.concatenate([db_tp, q_tp], axis=1), [119, 172, 48])
        tp_pairs.append(pair_img)

    # Concatenate all TP pairs vertically
    img = np.concatenate(tp_pairs, axis=0)

    # false positive FP
    if len(FP) > 0:
        # Limit to available FPs
        n_wrong = min(n_wrong, len(FP))
        idx_fp = np.random.permutation(len(FP))[:n_wrong]

        print(f'  Selected {n_wrong} wrong match(es) from {len(FP)} available')

        fp_pairs = []
        for i, idx in enumerate(idx_fp):
            db_idx = int(FP[idx, 0])
            q_idx = int(FP[idx, 1])
            print(f'    ✗ Wrong match {i+1}: DB image {db_idx} <-> Query image {q_idx}')

            db_fp = db_imgs[db_idx]
            q_fp = q_imgs[q_idx]

            if db_fp.shape != q_fp.shape:
                q_fp = resize(q_fp.copy(), db_fp.shape, anti_aliasing=True)
                q_fp = np.uint8(q_fp*255)

            pair_img = add_frame(np.concatenate([db_fp, q_fp], axis=1), [162, 20, 47])
            fp_pairs.append(pair_img)

        # Concatenate all FP pairs vertically
        img_fp = np.concatenate(fp_pairs, axis=0)
        img = np.concatenate([img, img_fp], axis=0)
    else:
        print(f'  No false positives available to display')

    # concat M
    if M is not None:
        M = resize(M.copy(), (img.shape[0], img.shape[0]))
        M = np.uint8(M.astype('float32')*255)
        M = np.tile(np.expand_dims(M, -1), (1, 1, 3))
        img = np.concatenate([M, img], axis=1)

    # show
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.imshow(img)
    ax.axis('off')
    plt.title(f'Examples: {n_correct} correct (green) and {n_wrong} wrong (red) matches from S>=thresh')
    plt.tight_layout()

    # Make sure the figure is drawn
    fig.canvas.draw()
    plt.draw()

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved matches visualization to: {save_path}')
        print(f'  Image shape: {img.shape}, dtype: {img.dtype}')
