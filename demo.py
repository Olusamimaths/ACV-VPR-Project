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

import argparse
import sys

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from evaluation.preprocessing import apply_preprocessing, preprocess_summary, preprocessing_suffix
from evaluation.run_output import DEFAULT_OUTPUT_ROOT, ExperimentRunOutput
from matching import matching
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset
from feature_extraction.factory import (
    PATCH_DESCRIPTOR_NAMES,
    PAIRWISE_DISTANCE_DESCRIPTOR_NAMES,
    SUPPORTED_DESCRIPTORS,
    create_feature_extractor,
)
import numpy as np

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=SUPPORTED_DESCRIPTORS, help='Select descriptor (default: HDC-DELF)')
    parser.add_argument('--dataset', type=str, default='GardensPoint', choices=['GardensPoint', 'StLucia', 'SFU'], help='Select dataset (default: GardensPoint)')
    parser.add_argument('--n_correct', type=int, default=1, help='Number of correct matches to display (default: 1)')
    parser.add_argument('--n_wrong', type=int, default=1, help='Number of wrong matches to display (default: 1)')
    parser.add_argument('--save_results', action='store_true', help='Save visualization results to output_images/')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT, help='Base directory for saved results (default: output_images/)')
    parser.add_argument('--vprtempo_model_path', type=str, default=None, help='Checkpoint path for VPRTempo (required when descriptor=VPRTempo)')
    parser.add_argument('--vprtempo_dims', type=str, default='56,56', help="Input dims for VPRTempo preprocessing, e.g. '56,56'")
    parser.add_argument('--vprtempo_patches', type=int, default=15, help='Patch normalization window used by VPRTempo (default: 15)')
    parser.add_argument('--vprtempo_batch_size', type=int, default=8, help='Batch size for VPRTempo feature extraction (default: 8)')
    parser.add_argument('--preprocess', type=str, default='none', choices=['none', 'clahe_query', 'clahe_all'], help='Optional image preprocessing before descriptor extraction')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--clahe_tile_grid', type=int, default=8, help='CLAHE tile grid size (default: 8)')
    args = parser.parse_args()

    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))
    print(f'========== Preprocessing: {preprocess_summary(args.preprocess, clip_limit=args.clahe_clip_limit, tile_grid_size=args.clahe_tile_grid)}')

    preprocess_suffix = preprocessing_suffix(args.preprocess)
    legacy_prefix = f'{args.dataset}{preprocess_suffix}'

    run_output = None
    if args.save_results:
        run_output = ExperimentRunOutput.create(
            args.output_root,
            run_slug=f'demo_{args.dataset}_{args.descriptor}{preprocess_suffix}',
            category='benchmarks',
        )
        print(f'===== Saving run outputs to {run_output.run_dir}')

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()
    imgs_db, imgs_q = apply_preprocessing(
        imgs_db,
        imgs_q,
        mode=args.preprocess,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid,
    )

    extractor_kwargs = {}
    if args.descriptor == 'VPRTempo':
        extractor_kwargs = {
            'vprtempo_model_path': args.vprtempo_model_path,
            'vprtempo_dims': args.vprtempo_dims,
            'vprtempo_patches': args.vprtempo_patches,
            'vprtempo_batch_size': args.vprtempo_batch_size,
        }
    feature_extractor = create_feature_extractor(args.descriptor, **extractor_kwargs)

    if args.descriptor not in PATCH_DESCRIPTOR_NAMES | PAIRWISE_DISTANCE_DESCRIPTOR_NAMES:
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)
        S = np.matmul(db_D_holistic , q_D_holistic.transpose())
    elif args.descriptor in PAIRWISE_DISTANCE_DESCRIPTOR_NAMES:
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # compute similarity matrix S with sum of absolute differences (SAD)
        print('===== Compute similarities S from sum of absolute differences (SAD)')
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i]-q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i,j] = -np.sum(np.abs(diff)) / dim
    else:
        print('=== WARNING: The PatchNetVLAD code in this repository is not optimised and will be slow and memory consuming.')
        print('===== Compute reference set descriptors')
        db_D_holigstic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        # S_hol = np.matmul(db_D_holistic , q_D_holistic.transpose())
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # show similarity matrix
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')
    if run_output is not None:
        run_output.savefig(fig, 'similarity_matrix.png', legacy_filename=f'{legacy_prefix}_similarity_matrix.png')

    # matching decision making
    print('===== Match images')

    # best match per query -> Single-best-match VPR
    M1 = matching.best_match_per_query(S)

    # thresholding -> Multi-match VPR
    M2 = matching.thresholding(S, 'auto')
    TP = np.argwhere(M2 & GThard)  # true positives
    FP = np.argwhere(M2 & ~GTsoft)  # false positives

    # evaluation
    print('===== Evaluation')
    # show correct and wrong image matches
    save_matches_path = run_output.run_path('matches_examples.png') if run_output is not None else None
    show_correct_and_wrong_matches.show(
        imgs_db, imgs_q, TP, FP,
        n_correct=args.n_correct,
        n_wrong=args.n_wrong,
        save_path=save_matches_path
    )
    if run_output is not None and save_matches_path is not None:
        try:
            run_output.copy_to_legacy(save_matches_path, legacy_filename=f'{legacy_prefix}_matches_examples.png')
        except FileNotFoundError:
            pass

    # show M's
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(M1)
    ax1.axis('off')
    ax1.set_title('Best match per query')
    ax2 = fig.add_subplot(122)
    ax2.imshow(M2)
    ax2.axis('off')
    ax2.set_title('Thresholding S>=thresh')
    if run_output is not None:
        run_output.savefig(fig, 'matching_results.png', legacy_filename=f'{legacy_prefix}_matching_results.png')

    # PR-curve
    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)
    plt.figure()
    plt.plot(R, P)
    plt.xlim(0, 1), plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Result on GardensPoint day_right--night_right')
    plt.grid('on')
    plt.draw()
    if run_output is not None:
        run_output.savefig(plt.gcf(), 'pr_curve.png', legacy_filename=f'{legacy_prefix}_pr_curve.png')

    # area under curve (AUC)
    AUC = np.trapz(P, R)
    print(f'\n===== AUC (area under curve): {AUC:.3f}')

    # maximum recall at 100% precision
    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'\n===== R@100P (maximum recall at 100% precision): {maxR:.2f}')

    # recall at K
    RatK = {}
    for K in [1, 5, 10]:
        RatK[K] = recallAtK(S, GThard, K=K)

    print(f'\n===== recall@K (R@K) -- R@1: {RatK[1]:.3f}, R@5: {RatK[5]:.3f}, R@10: {RatK[10]:.3f}')

    if run_output is not None:
        results_text = (
            'VPR Benchmark Test Results\n'
            + '=' * 70 + '\n\n'
            + f'Command: {" ".join(sys.argv)}\n'
            + f'Dataset: {args.dataset}\n'
            + f'Descriptor: {args.descriptor}\n'
            + f'Preprocessing: {preprocess_summary(args.preprocess, clip_limit=args.clahe_clip_limit, tile_grid_size=args.clahe_tile_grid)}\n'
            + f'Database images: {len(imgs_db)}\n'
            + f'Query images: {len(imgs_q)}\n'
            + f'True positives (thresholded): {len(TP)}\n'
            + f'False positives (thresholded): {len(FP)}\n\n'
            + f'AUC: {AUC:.3f}\n'
            + f'R@100P: {maxR:.3f}\n'
            + f'R@1: {RatK[1]:.3f}\n'
            + f'R@5: {RatK[5]:.3f}\n'
            + f'R@10: {RatK[10]:.3f}\n'
        )
        run_output.write_text('results.txt', results_text, legacy_filename=f'{legacy_prefix}_results.txt')

    plt.show()


if __name__ == "__main__":
    main()
