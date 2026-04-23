#!/usr/bin/env python3
"""
VPRTempo Demo - Get Actual Results

This script demonstrates VPRTempo integration with real images:
1. Build a reference map from GardensPoint day_left images
2. Query with day_right images
3. Display matching results and statistics

Usage:
    python demo_vprtempo_results.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple

def get_image_paths(directory: str) -> List[str]:
    """Get all image paths from directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
    image_paths = []
    for file in sorted(os.listdir(directory)):
        if Path(file).suffix.lower() in valid_extensions:
            image_paths.append(os.path.join(directory, file))
    return image_paths


def load_rgb_image(path: str, target_size: Tuple[int, int] = (480, 640)) -> np.ndarray:
    """Load and resize RGB image."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL not installed. Install with: pip install pillow")
        sys.exit(1)
    
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def demo_without_vprtempo():
    """Demo using mock extractors (no VPRTempo needed)."""
    print("\n" + "="*70)
    print("VPRTempo Demo - Mock Mode (VPRTempo not installed)")
    print("="*70)
    
    # Create mock descriptors
    np.random.seed(42)
    
    ref_dir = "images/GardensPoint/day_left"
    query_dir = "images/GardensPoint/day_right"
    
    ref_paths = get_image_paths(ref_dir)
    query_paths = get_image_paths(query_dir)
    
    print(f"\n📁 Reference images (day_left): {len(ref_paths)}")
    print(f"📁 Query images (day_right): {len(query_paths)}")
    
    # Simulate reference descriptors (256-dim, normalized)
    ref_descriptors = np.random.randn(len(ref_paths), 256).astype(np.float32)
    ref_descriptors /= np.linalg.norm(ref_descriptors, axis=1, keepdims=True)
    
    print(f"\n📊 Reference descriptors: shape {ref_descriptors.shape}")
    
    # Simulate query descriptors (correlated with references)
    query_descriptors = []
    for i, query_path in enumerate(query_paths):
        # Create query that's somewhat similar to a reference
        ref_idx = min(i, len(ref_paths) - 1)
        noise = np.random.randn(256).astype(np.float32) * 0.3
        query_desc = ref_descriptors[ref_idx] + noise
        query_desc /= np.linalg.norm(query_desc) + 1e-8
        query_descriptors.append(query_desc)
    
    query_descriptors = np.array(query_descriptors)
    print(f"📊 Query descriptors: shape {query_descriptors.shape}")
    
    # Compute similarities
    print("\n" + "-"*70)
    print("Localization Results:")
    print("-"*70)
    
    correct_matches = 0
    all_scores = []
    
    for q_idx, query_desc in enumerate(query_descriptors):
        # Cosine similarity
        similarities = (ref_descriptors @ query_desc).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        top_k_indices = np.argsort(similarities)[-3:][::-1]
        
        all_scores.append(best_score)
        
        # Check if correct (query_idx should match ref_idx)
        is_correct = (best_idx == q_idx)
        if is_correct:
            correct_matches += 1
        
        status = "✅ MATCH" if is_correct else "❌ MISMATCH"
        
        if q_idx < 5 or q_idx >= len(query_descriptors) - 2:  # Show first 5 and last 2
            print(f"Query {q_idx:3d}: Matched ref {best_idx:3d} (score: {best_score:.4f}) {status}")
        elif q_idx == 5:
            print("  ... (middle results omitted) ...")
    
    # Statistics
    recall_1 = correct_matches / len(query_descriptors)
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    print("\n" + "-"*70)
    print("Statistics:")
    print("-"*70)
    print(f"✅ Correct Matches (R@1): {correct_matches}/{len(query_descriptors)} ({recall_1*100:.1f}%)")
    print(f"📊 Mean Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"📊 Score Range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
    
    return True


def demo_with_vprtempo():
    """Demo using actual VPRTempo if available."""
    try:
        from vprtempo import VPRTempo
        print("\n✅ VPRTempo detected! Running with actual model...")
    except ImportError:
        print("⚠️  VPRTempo not installed. Install with: pip install vprtempo")
        return False
    
    print("\n" + "="*70)
    print("VPRTempo Demo - Real Model Mode")
    print("="*70)
    
    try:
        from live_vpr.extractors import create_feature_extractor, compute_global_descriptors
        from live_vpr.database import normalize_descriptors
        
        # Setup paths
        ref_dir = "images/GardensPoint/day_left"
        query_dir = "images/GardensPoint/day_right"
        
        ref_paths = get_image_paths(ref_dir)[:10]  # Limit for demo
        query_paths = get_image_paths(query_dir)[:10]
        
        print(f"\n📁 Reference images: {len(ref_paths)}")
        print(f"📁 Query images: {len(query_paths)}")
        
        # Load images
        print("\n⏳ Loading reference images...")
        ref_images = [load_rgb_image(path) for path in ref_paths]
        
        print("⏳ Loading query images...")
        query_images = [load_rgb_image(path) for path in query_paths]
        
        # Create extractor
        print("\n⏳ Creating VPRTempo extractor...")
        extractor = create_feature_extractor("VPRTempo")
        print(f"✅ Using device: {extractor.device}")
        
        # Extract descriptors
        print("⏳ Extracting reference descriptors...")
        ref_descriptors = compute_global_descriptors(extractor, ref_images)
        ref_descriptors = normalize_descriptors(ref_descriptors)
        print(f"✅ Shape: {ref_descriptors.shape}")
        
        print("⏳ Extracting query descriptors...")
        query_descriptors = compute_global_descriptors(extractor, query_images)
        query_descriptors = normalize_descriptors(query_descriptors)
        print(f"✅ Shape: {query_descriptors.shape}")
        
        # Compute similarities and get results
        print("\n" + "-"*70)
        print("Localization Results:")
        print("-"*70)
        
        correct_matches = 0
        all_scores = []
        
        for q_idx, query_desc in enumerate(query_descriptors):
            similarities = (ref_descriptors @ query_desc).flatten()
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            all_scores.append(best_score)
            
            is_correct = (best_idx == q_idx)
            if is_correct:
                correct_matches += 1
            
            status = "✅ MATCH" if is_correct else "❌ MISMATCH"
            print(f"Query {q_idx}: Matched ref {best_idx} (score: {best_score:.4f}) {status}")
        
        # Statistics
        print("\n" + "-"*70)
        print("Statistics:")
        print("-"*70)
        recall_1 = correct_matches / len(query_descriptors)
        print(f"✅ Recall@1: {correct_matches}/{len(query_descriptors)} ({recall_1*100:.1f}%)")
        print(f"📊 Mean Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run demo."""
    print("\n" + "="*70)
    print("VPRTempo Integration - Results Demo")
    print("="*70)
    
    # Try with actual VPRTempo first
    if demo_with_vprtempo():
        return 0
    
    # Fall back to mock demo
    print("\n💡 Switching to mock mode for demonstration...")
    try:
        demo_without_vprtempo()
        return 0
    except Exception as e:
        print(f"❌ Error in mock demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
