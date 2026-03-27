#!/usr/bin/env python3
#   =====================================================================
#   Campus Dataset VPR Test Script
#   Authors: Rhoda Ojetola, Peter Adeyemo, Samuel Olusola
#   Date: March 2026
#
#   Tests Visual Place Recognition on custom campus day/night dataset
#   =====================================================================
#
import argparse
import configparser
import os
import sys

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import CampusDataset
import numpy as np

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Visual Place Recognition test on custom campus dataset')
    parser.add_argument('--descriptor', type=str, default='CosPlace',
                       choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD',
                               'CosPlace', 'EigenPlaces', 'SAD'],
                       help='Select descriptor (default: CosPlace)')
    parser.add_argument('--dataset_dir', type=str, default='custom_dataset/',
                       help='Path to campus dataset directory')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--n_correct', type=int, default=3,
                       help='Number of correct matches to display (default: 3)')
    parser.add_argument('--n_wrong', type=int, default=5,
                       help='Number of wrong matches to display (default: 5)')
    args = parser.parse_args()

    print('=' * 70)
    print(f'Campus VPR Test: {args.descriptor} descriptor')
    print('=' * 70)

    # Load campus dataset
    print('\n===== Load campus dataset (day -> night)')
    dataset = CampusDataset(destination=args.dataset_dir)
    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    print(f'\n  Database (day images): {len(imgs_db)} images')
    print(f'  Queries (night images): {len(imgs_q)} images')
    print(f'  Images with ground truth matches: {np.sum(GThard.any(axis=0))}')
    print(f'  Images without matches (-npm): {np.sum(~GThard.any(axis=0))}')

    # Load feature extractor
    print(f'\n===== Load {args.descriptor} feature extractor')
    if args.descriptor == 'HDC-DELF':
        from feature_extraction.feature_extractor_holistic import HDCDELF
        feature_extractor = HDCDELF()
    elif args.descriptor == 'AlexNet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'SAD':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    elif args.descriptor == 'NetVLAD' or args.descriptor == 'PatchNetVLAD':
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        if args.descriptor == 'NetVLAD':
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
        else:
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        feature_extractor = PatchNetVLADFeatureExtractor(config)
    elif args.descriptor == 'CosPlace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'EigenPlaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)

    # Extract features and compute similarity matrix
    if args.descriptor != 'PatchNetVLAD' and args.descriptor != 'SAD':
        print('\n===== Compute database descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # Normalize and compute cosine similarity
        print('===== Compute cosine similarity matrix S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic, axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic, axis=1, keepdims=True)
        S = np.matmul(db_D_holistic, q_D_holistic.transpose())

    elif args.descriptor == 'SAD':
        print('\n===== Compute database descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # Compute sum of absolute differences
        print('===== Compute similarity matrix S (SAD)')
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i] - q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i, j] = -np.sum(np.abs(diff)) / dim

    else:  # PatchNetVLAD
        print('\n=== WARNING: PatchNetVLAD may be slow and memory consuming.')
        print('===== Compute database descriptors')
        db_D_holistic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print('===== Compute query descriptors')
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # Visualize similarity matrix
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(S, aspect='auto')
    plt.colorbar(label='Similarity')
    plt.xlabel('Query images (night)')
    plt.ylabel('Database images (day)')
    plt.title(f'Similarity Matrix S - {args.descriptor}')
    plt.tight_layout()
    if args.save_results:
        plt.savefig('output_images/campus_similarity_matrix.png', dpi=150)

    # Matching strategies
    print('\n===== Apply matching strategies')

    # Single best match per query
    M1 = matching.best_match_per_query(S)

    # Multi-match with automatic thresholding
    M2 = matching.thresholding(S, 'auto')

    # Find true positives and false positives
    TP = np.argwhere(M2 & GThard)  # true positives
    FP = np.argwhere(M2 & ~GTsoft)  # false positives

    print(f'  True positives (TP): {len(TP)}')
    print(f'  False positives (FP): {len(FP)}')

    # Visualize matches
    print('\n===== Visualize correct and wrong matches')
    if len(TP) > 0 or len(FP) > 0:
        save_matches_path = 'output_images/campus_matches_examples.png' if args.save_results else None
        show_correct_and_wrong_matches.show(
            imgs_db, imgs_q, TP, FP,
            n_correct=args.n_correct,
            n_wrong=args.n_wrong,
            save_path=save_matches_path
        )
        print(f'Displaying {min(args.n_correct, len(TP))} correct and {min(args.n_wrong, len(FP))} wrong matches')
    else:
        print('No matches to display')

    # Show matching matrices
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(M1, aspect='auto')
    ax1.set_xlabel('Query (night)')
    ax1.set_ylabel('Database (day)')
    ax1.set_title('Best match per query')
    ax1.grid(False)

    ax2 = fig.add_subplot(122)
    ax2.imshow(M2, aspect='auto')
    ax2.set_xlabel('Query (night)')
    ax2.set_ylabel('Database (day)')
    ax2.set_title('Multi-match (thresholding)')
    ax2.grid(False)
    plt.tight_layout()
    if args.save_results:
        plt.savefig('output_images/campus_matching_results.png', dpi=150)

    # Evaluation metrics
    print('\n' + '=' * 70)
    print('EVALUATION RESULTS')
    print('=' * 70)

    # Precision-Recall curve
    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(R, P, 'b-', linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Campus Dataset\n{args.descriptor}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save_results:
        plt.savefig('output_images/campus_pr_curve.png', dpi=150)

    # Area under curve
    AUC = np.trapz(P, R)
    print(f'\nAUC (Area Under PR Curve): {AUC:.3f}')

    # Maximum recall at 100% precision
    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'R@100P (Max Recall at 100% Precision): {maxR:.3f}')

    # Recall@K
    print('\nRecall@K:')
    RatK = {}
    for K in [1, 5, 10]:
        RatK[K] = recallAtK(S, GThard, K=K)
        print(f'  R@{K}: {RatK[K]:.3f}')

    # Additional analysis for -npm images
    print('\n' + '-' * 70)
    print('Analysis of images without perfect matches (-npm):')
    print('-' * 70)

    # Find queries without ground truth
    no_match_queries = np.where(~GThard.any(axis=0))[0]
    print(f'Number of -npm queries: {len(no_match_queries)}')

    if len(no_match_queries) > 0:
        print('\nTop matches for -npm queries (should ideally have low similarity):')
        for q_idx in no_match_queries[:5]:  # Show first 5
            best_db_idx = np.argmax(S[:, q_idx])
            best_sim = S[best_db_idx, q_idx]
            print(f'  Query {q_idx}: best match is DB {best_db_idx} with similarity {best_sim:.3f}')

    # Save results summary
    if args.save_results:
        results_file = 'output_images/campus_results.txt'
        with open(results_file, 'w') as f:
            f.write('Campus Dataset VPR Test Results\n')
            f.write('=' * 70 + '\n\n')
            f.write(f'Descriptor: {args.descriptor}\n')
            f.write(f'Database images: {len(imgs_db)}\n')
            f.write(f'Query images: {len(imgs_q)}\n')
            f.write(f'Queries with matches: {np.sum(GThard.any(axis=0))}\n')
            f.write(f'Queries without matches (-npm): {np.sum(~GThard.any(axis=0))}\n\n')
            f.write(f'AUC: {AUC:.3f}\n')
            f.write(f'R@100P: {maxR:.3f}\n')
            f.write(f'R@1: {RatK[1]:.3f}\n')
            f.write(f'R@5: {RatK[5]:.3f}\n')
            f.write(f'R@10: {RatK[10]:.3f}\n')
        print(f'\nResults saved to {results_file}')

    print('\n' + '=' * 70)
    print('Test complete! Close the plot windows to exit.')
    print('=' * 70)

    plt.show()


if __name__ == "__main__":
    main()
