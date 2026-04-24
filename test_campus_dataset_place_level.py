#!/usr/bin/env python3
#   =====================================================================
#   Campus Dataset VPR Test Script (Place-Level)
#   Companion script to the original 1-to-1 campus evaluation.
#   =====================================================================
#
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt

from datasets.load_dataset_place_level import CampusPlaceLevelDataset
from evaluation import show_correct_and_wrong_matches
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation.run_output import DEFAULT_OUTPUT_ROOT, ExperimentRunOutput
from feature_extraction.factory import (
    PATCH_DESCRIPTOR_NAMES,
    PAIRWISE_DISTANCE_DESCRIPTOR_NAMES,
    SUPPORTED_DESCRIPTORS,
    create_feature_extractor,
)
from matching import matching


def main():
    parser = argparse.ArgumentParser(
        description="Visual Place Recognition test on place-level rematched campus dataset"
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        default="CosPlace",
        choices=SUPPORTED_DESCRIPTORS,
        help="Select descriptor (default: CosPlace)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="custom_dataset_place_level/",
        help="Path to place-level campus dataset directory",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save plots and results to file",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory for saved results (default: output_images/)",
    )
    parser.add_argument(
        "--n_correct",
        type=int,
        default=3,
        help="Number of correct matches to display (default: 3)",
    )
    parser.add_argument(
        "--n_wrong",
        type=int,
        default=5,
        help="Number of wrong matches to display (default: 5)",
    )
    parser.add_argument(
        "--vprtempo_model_path",
        type=str,
        default=None,
        help="Checkpoint path for VPRTempo (required when descriptor=VPRTempo)",
    )
    parser.add_argument(
        "--vprtempo_dims",
        type=str,
        default="56,56",
        help="Input dims for VPRTempo preprocessing, e.g. '56,56'",
    )
    parser.add_argument(
        "--vprtempo_patches",
        type=int,
        default=15,
        help="Patch normalization window used by VPRTempo (default: 15)",
    )
    parser.add_argument(
        "--vprtempo_batch_size",
        type=int,
        default=8,
        help="Batch size for VPRTempo feature extraction (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Campus VPR Test (Place-Level): {args.descriptor} descriptor")
    print("=" * 70)

    run_output = None
    if args.save_results:
        run_output = ExperimentRunOutput.create(
            args.output_root,
            run_slug=f"campus_place_level_{args.descriptor}",
            category="campus-place-level",
        )
        print(f"\n===== Saving run outputs to {run_output.run_dir}")

    print("\n===== Load campus dataset (day -> night, place-level)")
    dataset = CampusPlaceLevelDataset(destination=args.dataset_dir)
    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    print(f"\n  Database (day images): {len(imgs_db)} images")
    print(f"  Queries (night images): {len(imgs_q)} images")
    print(f"  Queries with place-level matches: {np.sum(GThard.any(axis=0))}")
    print(f"  Queries without place-level matches: {np.sum(~GThard.any(axis=0))}")
    if dataset.summary is not None:
        print(f"  Queries rematched from original -npm labels: {dataset.summary.rematched_from_npm}")
        print(f"  Unique place IDs: {dataset.summary.unique_places}")

    print(f"\n===== Load {args.descriptor} feature extractor")
    extractor_kwargs = {}
    if args.descriptor == "VPRTempo":
        extractor_kwargs = {
            "vprtempo_model_path": args.vprtempo_model_path,
            "vprtempo_dims": args.vprtempo_dims,
            "vprtempo_patches": args.vprtempo_patches,
            "vprtempo_batch_size": args.vprtempo_batch_size,
        }
    feature_extractor = create_feature_extractor(args.descriptor, **extractor_kwargs)

    if args.descriptor not in PATCH_DESCRIPTOR_NAMES | PAIRWISE_DISTANCE_DESCRIPTOR_NAMES:
        print("\n===== Compute database descriptors")
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print("===== Compute query descriptors")
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        print("===== Compute cosine similarity matrix S")
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic, axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic, axis=1, keepdims=True)
        S = np.matmul(db_D_holistic, q_D_holistic.transpose())

    elif args.descriptor in PAIRWISE_DISTANCE_DESCRIPTOR_NAMES:
        print("\n===== Compute database descriptors")
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print("===== Compute query descriptors")
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        print("===== Compute similarity matrix S (SAD)")
        S = np.empty([len(imgs_db), len(imgs_q)], "float32")
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i] - q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i, j] = -np.sum(np.abs(diff)) / dim

    else:
        print("\n=== WARNING: PatchNetVLAD may be slow and memory consuming.")
        print("===== Compute database descriptors")
        db_D_holistic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print("===== Compute query descriptors")
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(S, aspect="auto")
    plt.colorbar(label="Similarity")
    plt.xlabel("Query images (night)")
    plt.ylabel("Database images (day)")
    plt.title(f"Similarity Matrix S - Place-Level - {args.descriptor}")
    plt.tight_layout()
    if run_output is not None:
        run_output.savefig(
            fig,
            "similarity_matrix.png",
            legacy_filename="campus_place_level_similarity_matrix.png",
        )

    print("\n===== Apply matching strategies")
    M1 = matching.best_match_per_query(S)
    M2 = matching.thresholding(S, "auto")

    TP = np.argwhere(M2 & GThard)
    FP = np.argwhere(M2 & ~GTsoft)

    print(f"  True positives (TP): {len(TP)}")
    print(f"  False positives (FP): {len(FP)}")

    print("\n===== Visualize correct and wrong matches")
    if len(TP) > 0 or len(FP) > 0:
        save_matches_path = run_output.run_path("matches_examples.png") if run_output is not None else None
        show_correct_and_wrong_matches.show(
            imgs_db,
            imgs_q,
            TP,
            FP,
            n_correct=args.n_correct,
            n_wrong=args.n_wrong,
            save_path=save_matches_path,
        )
        if run_output is not None and save_matches_path is not None:
            try:
                run_output.copy_to_legacy(
                    save_matches_path,
                    legacy_filename="campus_place_level_matches_examples.png",
                )
            except FileNotFoundError:
                pass
        print(f"Displaying {min(args.n_correct, len(TP))} correct and {min(args.n_wrong, len(FP))} wrong matches")
    else:
        print("No matches to display")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(M1, aspect="auto")
    ax1.set_xlabel("Query (night)")
    ax1.set_ylabel("Database (day)")
    ax1.set_title("Best match per query")
    ax1.grid(False)

    ax2 = fig.add_subplot(122)
    ax2.imshow(M2, aspect="auto")
    ax2.set_xlabel("Query (night)")
    ax2.set_ylabel("Database (day)")
    ax2.set_title("Multi-match (thresholding)")
    ax2.grid(False)
    plt.tight_layout()
    if run_output is not None:
        run_output.savefig(
            fig,
            "matching_results.png",
            legacy_filename="campus_place_level_matching_results.png",
        )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    P, R = createPR(S, GThard, GTsoft, matching="multi", n_thresh=100)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(R, P, "b-", linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - Campus Place-Level Dataset\n{args.descriptor}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if run_output is not None:
        run_output.savefig(
            fig,
            "pr_curve.png",
            legacy_filename="campus_place_level_pr_curve.png",
        )

    AUC = np.trapz(P, R)
    print(f"\nAUC (Area Under PR Curve): {AUC:.3f}")

    maxR = recallAt100precision(S, GThard, GTsoft, matching="multi", n_thresh=100)
    print(f"R@100P (Max Recall at 100% Precision): {maxR:.3f}")

    print("\nRecall@K:")
    RatK = {}
    for K in [1, 5, 10]:
        RatK[K] = recallAtK(S, GThard, K=K)
        print(f"  R@{K}: {RatK[K]:.3f}")

    print("\n" + "-" * 70)
    print("Analysis of remaining no-match queries:")
    print("-" * 70)
    no_match_queries = np.where(~GThard.any(axis=0))[0]
    print(f"Number of queries without place-level matches: {len(no_match_queries)}")

    if len(no_match_queries) > 0:
        print("\nTop matches for remaining no-match queries:")
        for q_idx in no_match_queries[:5]:
            best_db_idx = np.argmax(S[:, q_idx])
            best_sim = S[best_db_idx, q_idx]
            print(f"  Query {q_idx}: best match is DB {best_db_idx} with similarity {best_sim:.3f}")

    if run_output is not None:
        results_text = (
            "Campus Dataset VPR Test Results (Place-Level)\n"
            + "=" * 70 + "\n\n"
            + f"Command: {' '.join(sys.argv)}\n"
            + f"Descriptor: {args.descriptor}\n"
            + f"Dataset directory: {args.dataset_dir}\n"
            + f"Database images: {len(imgs_db)}\n"
            + f"Query images: {len(imgs_q)}\n"
            + f"Queries with place-level matches: {np.sum(GThard.any(axis=0))}\n"
            + f"Queries without place-level matches: {np.sum(~GThard.any(axis=0))}\n"
            + f"True positives (thresholded): {len(TP)}\n"
            + f"False positives (thresholded): {len(FP)}\n"
        )
        if dataset.summary is not None:
            results_text += (
                f"Queries rematched from original -npm labels: {dataset.summary.rematched_from_npm}\n"
                + f"Unique place IDs: {dataset.summary.unique_places}\n"
            )
        results_text += (
            "\n"
            + f"AUC: {AUC:.3f}\n"
            + f"R@100P: {maxR:.3f}\n"
            + f"R@1: {RatK[1]:.3f}\n"
            + f"R@5: {RatK[5]:.3f}\n"
            + f"R@10: {RatK[10]:.3f}\n"
        )
        results_file = run_output.write_text(
            "results.txt",
            results_text,
            legacy_filename="campus_place_level_results.txt",
        )
        print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 70)
    print("Place-level test complete! Close the plot windows to exit.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
