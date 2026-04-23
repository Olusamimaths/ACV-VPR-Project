#!/usr/bin/env python3
"""
VPRTempo Pipeline on Campus Day/Night Dataset

This script runs the complete VPRTempo workflow on your custom campus dataset:
1. Build reference map from day images
2. Query with night images
3. Evaluate matching accuracy
4. Compare with other descriptors

Usage:
    python campus_vprtempo_workflow.py [--descriptor {VPRTempo|VPRTempoQuant|CosPlace|...}]
                                       [--use_temporal] [--temporal_window_size 5]
                                       [--save_results] [--compare]
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple
import time

# Add project to path
sys.path.insert(0, '/Users/rhoda/VPR_Tutorial')

from live_vpr.extractors import create_feature_extractor, compute_global_descriptors
from live_vpr.database import normalize_descriptors, ReferenceMap, save_reference_map, load_reference_map
from live_vpr.online import LiveLocalizer
from matching import matching
from evaluation.metrics import createPR, recallAt100precision, recallAtK

try:
    from PIL import Image
except ImportError:
    print("ERROR: PIL not installed. Install with: pip install pillow")
    sys.exit(1)

import matplotlib.pyplot as plt


class CampusVPRTempoWorkflow:
    """Complete VPRTempo workflow for campus day/night dataset"""
    
    def __init__(self, descriptor_name: str = "VPRTempo", 
                 use_temporal: bool = False,
                 temporal_window_size: int = 5):
        """Initialize workflow"""
        self.descriptor_name = descriptor_name
        self.use_temporal = use_temporal
        self.temporal_window_size = temporal_window_size
        self.dataset_dir = "custom_dataset"
        self.results_dir = "output_images"
        
        # Create results directory if needed
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"VPRTempo Campus Day/Night Workflow")
        print("="*70)
        print(f"Descriptor: {self.descriptor_name}")
        print(f"Temporal Mode: {self.use_temporal}")
        if self.use_temporal:
            print(f"Temporal Window Size: {self.temporal_window_size}")
        print("="*70)
    
    def load_images(self, directory: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load all images from directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
        images = []
        paths = []
        
        for filename in sorted(os.listdir(directory)):
            if Path(filename).suffix.lower() in valid_extensions:
                filepath = os.path.join(directory, filename)
                try:
                    img = Image.open(filepath).convert('RGB')
                    # Resize to standard size (480, 640)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    images.append(np.array(img, dtype=np.uint8))
                    paths.append(filepath)
                except Exception as e:
                    print(f"  ⚠️  Could not load {filename}: {e}")
        
        return images, paths
    
    def step1_build_reference_map(self) -> ReferenceMap:
        """Step 1: Build reference map from day images"""
        print("\n" + "-"*70)
        print("STEP 1: Build Reference Map from Day Images")
        print("-"*70)
        
        day_dir = os.path.join(self.dataset_dir, "day_images")
        
        # Load images
        print(f"\n⏳ Loading day images from: {day_dir}")
        day_images, day_paths = self.load_images(day_dir)
        print(f"✅ Loaded {len(day_images)} day images")
        
        # Create extractor
        print(f"\n⏳ Creating {self.descriptor_name} extractor...")
        try:
            extractor = create_feature_extractor(self.descriptor_name)
            print(f"✅ Extractor created (device: {getattr(extractor, 'device', 'unknown')})")
        except Exception as e:
            print(f"❌ Error creating extractor: {e}")
            print(f"   Make sure VPRTempo is installed: pip install vprtempo")
            raise
        
        # Extract descriptors
        print(f"\n⏳ Extracting descriptors from {len(day_images)} day images...")
        start = time.time()
        descriptors = compute_global_descriptors(extractor, day_images)
        elapsed = time.time() - start
        print(f"✅ Extracted {descriptors.shape[0]} descriptors (shape: {descriptors.shape})")
        print(f"   Time: {elapsed:.2f}s ({elapsed/len(day_images):.2f}s per image)")
        
        # Normalize descriptors
        print(f"\n⏳ Normalizing descriptors...")
        descriptors = normalize_descriptors(descriptors)
        print(f"✅ Normalized (mean norm: {np.mean(np.linalg.norm(descriptors, axis=1)):.4f})")
        
        # Create reference map
        print(f"\n⏳ Creating reference map...")
        reference_map = ReferenceMap(
            descriptors=descriptors,
            image_paths=day_paths,
            descriptor_name=self.descriptor_name,
            target_size=(640, 480),
            metadata={"dataset": "campus_day", "count": len(day_images)}
        )
        print(f"✅ Reference map created")
        
        return reference_map, day_images
    
    def step2_query_with_night_images(self, reference_map: ReferenceMap) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """Step 2: Query with night images and compute similarities"""
        print("\n" + "-"*70)
        print("STEP 2: Query with Night Images")
        print("-"*70)
        
        night_dir = os.path.join(self.dataset_dir, "night_images")
        
        # Load images
        print(f"\n⏳ Loading night images from: {night_dir}")
        night_images, night_paths = self.load_images(night_dir)
        print(f"✅ Loaded {len(night_images)} night images")
        
        # Create extractor
        print(f"\n⏳ Creating {self.descriptor_name} extractor for queries...")
        extractor = create_feature_extractor(self.descriptor_name)
        
        # Extract descriptors
        print(f"\n⏳ Extracting descriptors from {len(night_images)} night images...")
        start = time.time()
        query_descriptors = compute_global_descriptors(extractor, night_images)
        elapsed = time.time() - start
        print(f"✅ Extracted {query_descriptors.shape[0]} query descriptors")
        print(f"   Time: {elapsed:.2f}s ({elapsed/len(night_images):.2f}s per image)")
        
        # Normalize
        query_descriptors = normalize_descriptors(query_descriptors)
        
        # Compute similarities
        print(f"\n⏳ Computing cosine similarity matrix...")
        similarities = (reference_map.descriptors @ query_descriptors.T)
        print(f"✅ Similarity matrix shape: {similarities.shape}")
        
        return similarities, night_images, night_paths
    
    def step3_evaluate_matching(self, reference_map: ReferenceMap, 
                               similarities: np.ndarray,
                               day_images: List[np.ndarray],
                               night_images: List[np.ndarray]):
        """Step 3: Evaluate matching quality"""
        print("\n" + "-"*70)
        print("STEP 3: Evaluate Matching Quality")
        print("-"*70)
        
        # Matching strategies
        print(f"\n⏳ Applying matching strategies...")
        
        # Best match per query
        M_best = matching.best_match_per_query(similarities)
        best_matches = np.argmax(similarities, axis=0)
        best_scores = np.max(similarities, axis=0)
        
        print(f"\n📊 Best Match Per Query:")
        print(f"   Mean score: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")
        print(f"   Min score: {np.min(best_scores):.4f}")
        print(f"   Max score: {np.max(best_scores):.4f}")
        
        # Show sample matches
        print(f"\n📋 Sample Matches (first 10):")
        print(f"   Query → Database (score)")
        print(f"   {'─'*40}")
        for q_idx in range(min(10, len(night_images))):
            db_idx = best_matches[q_idx]
            score = best_scores[q_idx]
            status = "✅" if db_idx == q_idx else "❌"
            print(f"   Night {q_idx:2d} → Day {db_idx:2d} ({score:.4f}) {status}")
        
        # Visualize similarity matrix
        print(f"\n⏳ Creating similarity matrix visualization...")
        self._plot_similarity_matrix(similarities, "campus_vprtempo_similarity.png")
        
        # Statistics
        print(f"\n" + "-"*70)
        print("MATCHING STATISTICS:")
        print("-"*70)
        
        # Diagonal accuracy (perfect match rate)
        diagonal_matches = sum(best_matches[i] == i for i in range(len(best_matches)))
        diagonal_accuracy = diagonal_matches / len(best_matches)
        print(f"✅ Diagonal Matches (Day[i] ↔ Night[i]): {diagonal_matches}/{len(best_matches)} ({diagonal_accuracy*100:.1f}%)")
        
        # Top-K accuracy
        print(f"\n📊 Top-K Accuracy:")
        for k in [1, 3, 5]:
            top_k_matches = 0
            for q_idx in range(len(night_images)):
                true_match = q_idx
                top_k_indices = np.argsort(similarities[:, q_idx])[-k:][::-1]
                if true_match in top_k_indices:
                    top_k_matches += 1
            accuracy = top_k_matches / len(night_images)
            print(f"   Top-{k}: {accuracy*100:.1f}%")
        
        return best_matches, best_scores
    
    def step4_temporal_analysis(self, similarities: np.ndarray):
        """Step 4: Analyze temporal aggregation benefits (optional)"""
        if not self.use_temporal:
            return
        
        print("\n" + "-"*70)
        print("STEP 4: Temporal Aggregation Analysis")
        print("-"*70)
        
        print(f"\n⏳ Simulating temporal aggregation (window size: {self.temporal_window_size})...")
        
        # Simulate temporal aggregation
        temporal_accuracies = []
        baseline_accuracies = []
        
        for window_idx in range(self.temporal_window_size, len(similarities.T)):
            # Baseline: single frame
            best_idx_baseline = np.argmax(similarities[:, window_idx])
            baseline_correct = (best_idx_baseline == window_idx)
            baseline_accuracies.append(baseline_correct)
            
            # Temporal: aggregate window
            window_start = window_idx - self.temporal_window_size + 1
            window_sims = similarities[:, window_start:window_idx+1]
            
            # Weighted mean aggregation (recent frames weighted higher)
            weights = np.linspace(0.5, 1.0, self.temporal_window_size)
            weights /= weights.sum()
            aggregated_sims = np.average(window_sims, axis=1, weights=weights)
            
            best_idx_temporal = np.argmax(aggregated_sims)
            temporal_correct = (best_idx_temporal == window_idx)
            temporal_accuracies.append(temporal_correct)
        
        baseline_acc = np.mean(baseline_accuracies) * 100
        temporal_acc = np.mean(temporal_accuracies) * 100
        improvement = temporal_acc - baseline_acc
        
        print(f"\n📊 Temporal Aggregation Results:")
        print(f"   Baseline (frame-by-frame): {baseline_acc:.1f}%")
        print(f"   Temporal (window={self.temporal_window_size}): {temporal_acc:.1f}%")
        print(f"   Improvement: {improvement:+.1f}%")
    
    def step5_compare_with_other_descriptors(self, similarities_vprtempo: np.ndarray):
        """Step 5: Compare with other descriptors (optional)"""
        print("\n" + "-"*70)
        print("STEP 5: Comparing with Other Descriptors")
        print("-"*70)
        
        if self.descriptor_name in ["VPRTempo", "VPRTempoQuant"]:
            print("\n⏳ Comparing VPRTempo with CosPlace...")
            
            # Load dataset
            day_dir = os.path.join(self.dataset_dir, "day_images")
            night_dir = os.path.join(self.dataset_dir, "night_images")
            
            day_images, _ = self.load_images(day_dir)
            night_images, _ = self.load_images(night_dir)
            
            # Extract with CosPlace
            try:
                from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
                print("⏳ Extracting with CosPlace...")
                cosplace_extractor = CosPlaceFeatureExtractor()
                
                day_desc_cosplace = compute_global_descriptors(cosplace_extractor, day_images)
                day_desc_cosplace = normalize_descriptors(day_desc_cosplace)
                
                night_desc_cosplace = compute_global_descriptors(cosplace_extractor, night_images)
                night_desc_cosplace = normalize_descriptors(night_desc_cosplace)
                
                similarities_cosplace = (day_desc_cosplace @ night_desc_cosplace.T)
                
                # Compare accuracy
                vprtempo_best = np.argmax(similarities_vprtempo, axis=0)
                cosplace_best = np.argmax(similarities_cosplace, axis=0)
                
                vprtempo_diag = sum(vprtempo_best[i] == i for i in range(len(vprtempo_best)))
                cosplace_diag = sum(cosplace_best[i] == i for i in range(len(cosplace_best)))
                
                vprtempo_acc = vprtempo_diag / len(vprtempo_best) * 100
                cosplace_acc = cosplace_diag / len(cosplace_best) * 100
                
                print(f"\n📊 Comparison Results:")
                print(f"   VPRTempo: {vprtempo_acc:.1f}% diagonal accuracy")
                print(f"   CosPlace: {cosplace_acc:.1f}% diagonal accuracy")
                print(f"   Difference: {vprtempo_acc - cosplace_acc:+.1f}%")
                
            except Exception as e:
                print(f"⚠️  Could not compare with CosPlace: {e}")
    
    def _plot_similarity_matrix(self, S: np.ndarray, filename: str):
        """Plot and save similarity matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(S, aspect='auto', cmap='viridis')
        
        ax.set_xlabel('Query Images (Night)', fontsize=12)
        ax.set_ylabel('Reference Images (Day)', fontsize=12)
        ax.set_title(f'Similarity Matrix - {self.descriptor_name}\n(Day vs Night Campus Images)', 
                    fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Add diagonal line for reference
        n = min(S.shape)
        ax.plot([0, n], [0, n], 'r--', linewidth=2, alpha=0.5, label='Perfect matches')
        ax.legend()
        
        filepath = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        print(f"   ✅ Saved to: {filepath}")
        plt.close()
    
    def run(self):
        """Run complete workflow"""
        try:
            # Step 1: Build reference map
            reference_map, day_images = self.step1_build_reference_map()
            
            # Step 2: Query and compute similarities
            similarities, night_images, night_paths = self.step2_query_with_night_images(reference_map)
            
            # Step 3: Evaluate
            best_matches, best_scores = self.step3_evaluate_matching(
                reference_map, similarities, day_images, night_images
            )
            
            # Step 4: Temporal analysis (if enabled)
            if self.use_temporal:
                self.step4_temporal_analysis(similarities)
            
            # Step 5: Compare with other descriptors (if requested)
            # self.step5_compare_with_other_descriptors(similarities)
            
            print("\n" + "="*70)
            print("✅ Workflow Complete!")
            print("="*70)
            print(f"\nResults saved to: {self.results_dir}/")
            print("\n🎉 Campus VPRTempo evaluation finished successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during workflow: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run VPRTempo pipeline on campus day/night dataset"
    )
    parser.add_argument(
        '--descriptor',
        type=str,
        default='VPRTempo',
        choices=['VPRTempo', 'VPRTempoQuant', 'CosPlace', 'EigenPlaces', 'NetVLAD'],
        help='Descriptor to use (default: VPRTempo)'
    )
    parser.add_argument(
        '--use_temporal',
        action='store_true',
        help='Enable temporal aggregation'
    )
    parser.add_argument(
        '--temporal_window_size',
        type=int,
        default=5,
        help='Temporal window size (default: 5)'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save detailed results'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with other descriptors'
    )
    
    args = parser.parse_args()
    
    # Run workflow
    workflow = CampusVPRTempoWorkflow(
        descriptor_name=args.descriptor,
        use_temporal=args.use_temporal,
        temporal_window_size=args.temporal_window_size
    )
    
    return workflow.run()


if __name__ == "__main__":
    sys.exit(main())
