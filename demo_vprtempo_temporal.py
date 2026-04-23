#!/usr/bin/env python3
"""
VPRTempo Temporal Mode Demo

This script demonstrates temporal aggregation for improved accuracy:
1. Creates a sequence of queries
2. Shows frame-by-frame localization (no temporal)
3. Shows temporal aggregation localization
4. Compares the results and stability

Usage:
    python demo_vprtempo_temporal.py
"""

import numpy as np
import sys
from typing import List, Tuple


def demo_temporal_aggregation():
    """Demonstrate temporal aggregation benefits."""
    print("\n" + "="*70)
    print("VPRTempo Temporal Aggregation Demo")
    print("="*70)
    
    np.random.seed(42)
    
    # Create synthetic scenario
    num_refs = 100
    num_frames = 20
    
    # Reference descriptors
    ref_descriptors = np.random.randn(num_refs, 256).astype(np.float32)
    ref_descriptors /= np.linalg.norm(ref_descriptors, axis=1, keepdims=True)
    
    print(f"\n📊 Setup:")
    print(f"   Reference locations: {num_refs}")
    print(f"   Query sequence length: {num_frames}")
    print(f"   Descriptor dimension: 256")
    
    # Simulate a traversal: frames come from nearby locations
    # True location is location 5, but with noise
    true_location = 5
    query_frames = []
    for i in range(num_frames):
        # Create query descriptors that are noisy versions of true location
        noise = np.random.randn(256).astype(np.float32) * 0.4
        query_desc = ref_descriptors[true_location] + noise
        query_desc /= np.linalg.norm(query_desc) + 1e-8
        query_frames.append(query_desc)
    
    query_frames = np.array(query_frames)
    print(f"   True location: {true_location}")
    
    # --- FRAME-BY-FRAME LOCALIZATION ---
    print("\n" + "-"*70)
    print("Frame-by-Frame Localization (No Temporal):")
    print("-"*70)
    
    frame_by_frame_matches = []
    frame_by_frame_scores = []
    
    for frame_idx, query_desc in enumerate(query_frames):
        similarities = (ref_descriptors @ query_desc).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        frame_by_frame_matches.append(best_idx)
        frame_by_frame_scores.append(best_score)
        
        is_correct = (best_idx == true_location)
        status = "✅" if is_correct else "❌"
        print(f"Frame {frame_idx:2d}: Matched location {best_idx:3d} (score: {best_score:.4f}) {status}")
    
    frame_accuracy = sum(1 for m in frame_by_frame_matches if m == true_location) / num_frames
    print(f"\nFrame-by-frame Recall@1: {frame_accuracy*100:.1f}%")
    
    # --- TEMPORAL AGGREGATION ---
    print("\n" + "-"*70)
    print("Temporal Aggregation (Window Size = 5):")
    print("-"*70)
    
    window_size = 5
    temporal_buffer = []
    temporal_matches = []
    temporal_scores = []
    
    for frame_idx, query_desc in enumerate(query_frames):
        # Add to buffer
        temporal_buffer.append(query_desc)
        if len(temporal_buffer) > window_size:
            temporal_buffer.pop(0)
        
        # Use aggregated descriptor if buffer is full
        if len(temporal_buffer) == window_size:
            # Weighted mean: recent frames have more weight
            weights = np.linspace(0.5, 1.0, window_size)
            weights /= weights.sum()
            aggregated = np.average(temporal_buffer, axis=0, weights=weights)
            aggregated /= np.linalg.norm(aggregated) + 1e-8
        else:
            # Not ready yet, use current frame
            aggregated = query_desc
        
        similarities = (ref_descriptors @ aggregated).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        temporal_matches.append(best_idx)
        temporal_scores.append(best_score)
        
        is_correct = (best_idx == true_location)
        is_ready = len(temporal_buffer) == window_size
        status = "✅" if is_correct else "❌"
        ready = "📦" if is_ready else "⏳"
        
        print(f"Frame {frame_idx:2d}: Matched location {best_idx:3d} (score: {best_score:.4f}) {status} {ready}")
    
    temporal_accuracy = sum(1 for m in temporal_matches if m == true_location) / num_frames
    print(f"\nTemporal Recall@1: {temporal_accuracy*100:.1f}%")
    
    # --- COMPARISON ---
    print("\n" + "-"*70)
    print("Comparison:")
    print("-"*70)
    
    # Count improvements
    improved = sum(
        1 for i in range(num_frames)
        if temporal_matches[i] == true_location and frame_by_frame_matches[i] != true_location
    )
    degraded = sum(
        1 for i in range(num_frames)
        if temporal_matches[i] != true_location and frame_by_frame_matches[i] == true_location
    )
    
    print(f"Frame-by-frame Recall@1:  {frame_accuracy*100:5.1f}%")
    print(f"Temporal Recall@1:        {temporal_accuracy*100:5.1f}%")
    print(f"Improvement:              {(temporal_accuracy - frame_accuracy)*100:+5.1f}%")
    print()
    print(f"Frames improved by temporal:  {improved}")
    print(f"Frames degraded by temporal:  {degraded}")
    
    # Score stability
    frame_score_std = np.std(frame_by_frame_scores)
    temporal_score_std = np.std(temporal_scores)
    
    print()
    print(f"Frame-by-frame score stability (std): {frame_score_std:.4f}")
    print(f"Temporal score stability (std):       {temporal_score_std:.4f}")
    print(f"Temporal is {frame_score_std/temporal_score_std:.1f}x more stable")
    
    # Visualize sequence
    print("\n" + "-"*70)
    print("Visualization (first 15 frames):")
    print("-"*70)
    print("Legend: ✅ = correct, ❌ = wrong, | = frame-by-frame, ◆ = temporal")
    print()
    
    for i in range(min(15, num_frames)):
        fb_correct = frame_by_frame_matches[i] == true_location
        t_correct = temporal_matches[i] == true_location
        
        fb_char = "✅" if fb_correct else "❌"
        t_char = "✅" if t_correct else "❌"
        
        print(f"Frame {i:2d}: {fb_char} | {t_char} ◆")


def main():
    """Run demo."""
    print("\n" + "="*70)
    print("VPRTempo Temporal Processing Demo")
    print("="*70)
    
    try:
        demo_temporal_aggregation()
        print("\n" + "="*70)
        print("Demo Complete!")
        print("="*70)
        print("\n💡 Key Insights:")
        print("   • Temporal aggregation smooths noisy frame-by-frame matches")
        print("   • Improves accuracy especially in challenging conditions")
        print("   • Trade-off: ~1-2 frame latency for better results")
        print("   • Most effective in smooth camera motion")
        print("   • Can be toggled on/off based on needs")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
