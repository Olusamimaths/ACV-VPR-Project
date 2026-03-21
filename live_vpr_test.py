#!/usr/bin/env python3
"""
Live VPR Testing Script for CMU-Africa Campus Dataset
Authors: Rhoda Ojetola, Peter Adeyemo, Samuel Olusola
Date: March 2026

This script enables real-time Visual Place Recognition testing using
a phone camera connected to a Mac laptop.

Usage:
    # Step 1: Build reference database from day images
    python live_vpr_test.py --mode build_db --data_dir data/CMUAfrica/day --db_path day_database.npz

    # Step 2a: Run live test with camera
    python live_vpr_test.py --mode live_test --db_path day_database.npz --camera 0

    # Step 2b: Or test with recorded video
    python live_vpr_test.py --mode video_test --db_path day_database.npz --video path/to/video.mp4

    # Step 3: Batch evaluation (offline testing)
    python live_vpr_test.py --mode batch_eval --db_path day_database.npz --query_dir data/CMUAfrica/night

Controls (during live/video test):
    q - Quit
    s - Save current frame
    t - Toggle showing top-3 matches
    +/- - Adjust threshold
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class FeatureExtractorWrapper:
    """Wrapper to handle different feature extractors uniformly."""

    def __init__(self, descriptor_name='CosPlace'):
        self.descriptor_name = descriptor_name
        self.extractor = None
        self._load_extractor()

    def _load_extractor(self):
        """Load the specified feature extractor."""
        print(f"Loading {self.descriptor_name} model...")
        start = time.time()

        if self.descriptor_name == 'CosPlace':
            from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
            self.extractor = CosPlaceFeatureExtractor()
        elif self.descriptor_name == 'EigenPlaces':
            from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
            self.extractor = EigenPlacesFeatureExtractor()
        elif self.descriptor_name == 'NetVLAD':
            from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
            self.extractor = PatchNetVLADFeatureExtractor()
        elif self.descriptor_name == 'HDC-DELF':
            from feature_extraction.feature_extractor_holistic import HDCDELF
            self.extractor = HDCDELF()
        elif self.descriptor_name == 'AlexNet':
            from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
            self.extractor = AlexNetConv3Extractor()
        else:
            raise ValueError(f"Unknown descriptor: {self.descriptor_name}")

        print(f"Model loaded in {time.time()-start:.2f}s")

    def compute_features(self, images):
        """Extract features from a list of images."""
        return self.extractor.compute_features(images)


class LiveVPR:
    """
    Live Visual Place Recognition System.

    This class handles:
    - Building reference databases from image directories
    - Real-time localization from camera feed
    - Batch evaluation on recorded videos or image sets
    """

    def __init__(self, descriptor='CosPlace'):
        """
        Initialize the VPR system.

        Args:
            descriptor: Feature extractor to use
                       ('CosPlace', 'EigenPlaces', 'NetVLAD', 'HDC-DELF', 'AlexNet')
        """
        self.extractor = FeatureExtractorWrapper(descriptor)
        self.descriptor_name = descriptor
        self.D_db = None
        self.db_image_paths = None
        self.db_metadata = None

    def build_database(self, image_dir, output_path='database.npz',
                       target_size=(640, 480)):
        """
        Build reference database from images in a directory.

        Args:
            image_dir: Path to directory containing reference images
            output_path: Where to save the database file
            target_size: Resize images to this size (width, height)

        Returns:
            Tuple of (descriptors array, image paths list)
        """
        print(f"\n{'='*60}")
        print(f"BUILDING REFERENCE DATABASE")
        print(f"{'='*60}")
        print(f"Source directory: {image_dir}")
        print(f"Output file: {output_path}")
        print(f"Target size: {target_size}")
        print(f"Descriptor: {self.descriptor_name}")
        print(f"{'='*60}\n")

        # Find all images
        image_dir = Path(image_dir)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(ext))
        image_paths = sorted([str(p) for p in image_paths])

        if len(image_paths) == 0:
            print(f"ERROR: No images found in {image_dir}")
            return None, None

        print(f"Found {len(image_paths)} images")

        # Load and preprocess images
        print("Loading images...")
        images = []
        valid_paths = []
        for i, p in enumerate(image_paths):
            img = cv2.imread(p)
            if img is None:
                print(f"  Warning: Could not read {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
            valid_paths.append(p)

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i+1}/{len(image_paths)} images")

        print(f"Successfully loaded {len(images)} images")

        # Extract features
        print("\nExtracting features...")
        start = time.time()
        D_db = self.extractor.compute_features(images)
        extraction_time = time.time() - start
        print(f"Feature extraction took {extraction_time:.2f}s")
        print(f"Average: {extraction_time/len(images)*1000:.1f}ms per image")

        # Normalize descriptors
        norms = np.linalg.norm(D_db, axis=1, keepdims=True)
        D_db = D_db / (norms + 1e-8)

        # Create metadata
        metadata = {
            'descriptor': self.descriptor_name,
            'num_images': len(valid_paths),
            'descriptor_dim': D_db.shape[1],
            'target_size': target_size,
            'created': datetime.now().isoformat(),
            'source_dir': str(image_dir)
        }

        # Save database
        np.savez(output_path,
                 descriptors=D_db,
                 image_paths=np.array(valid_paths),
                 metadata=json.dumps(metadata))

        print(f"\n{'='*60}")
        print(f"DATABASE CREATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Saved to: {output_path}")
        print(f"Descriptor shape: {D_db.shape}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*60}\n")

        self.D_db = D_db
        self.db_image_paths = valid_paths
        self.db_metadata = metadata

        return D_db, valid_paths

    def load_database(self, db_path='database.npz'):
        """
        Load a pre-built database.

        Args:
            db_path: Path to the database file
        """
        print(f"Loading database from {db_path}...")

        if not os.path.exists(db_path):
            print(f"ERROR: Database file not found: {db_path}")
            return False

        data = np.load(db_path, allow_pickle=True)
        self.D_db = data['descriptors']
        self.db_image_paths = list(data['image_paths'])

        if 'metadata' in data:
            self.db_metadata = json.loads(str(data['metadata']))
            print(f"Database info:")
            print(f"  - Descriptor: {self.db_metadata.get('descriptor', 'unknown')}")
            print(f"  - Images: {self.db_metadata.get('num_images', len(self.db_image_paths))}")
            print(f"  - Dimension: {self.db_metadata.get('descriptor_dim', self.D_db.shape[1])}")
            print(f"  - Created: {self.db_metadata.get('created', 'unknown')}")
        else:
            print(f"Loaded {len(self.db_image_paths)} reference images")
            print(f"Descriptor dimension: {self.D_db.shape[1]}")

        return True

    def localize(self, image, top_k=5, threshold=0.5, target_size=(640, 480)):
        """
        Find matching location for a single image.

        Args:
            image: Input image (BGR format from OpenCV)
            top_k: Number of top matches to return
            threshold: Minimum similarity score for recognition
            target_size: Size to resize input image

        Returns:
            Dictionary with match results
        """
        # Preprocess image
        img = cv2.resize(image, target_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract feature
        start = time.time()
        d_q = self.extractor.compute_features([img_rgb])
        extraction_time = time.time() - start

        # Normalize
        d_q = d_q / (np.linalg.norm(d_q) + 1e-8)

        # Compute similarities
        similarities = (self.D_db @ d_q.T).flatten()

        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        return {
            'best_match_idx': int(top_indices[0]),
            'best_score': float(top_scores[0]),
            'top_k_indices': top_indices.tolist(),
            'top_k_scores': top_scores.tolist(),
            'recognized': float(top_scores[0]) > threshold,
            'extraction_time_ms': extraction_time * 1000,
            'all_similarities': similarities
        }

    def run_live(self, camera_id=0, threshold=0.5, show_top_k=True):
        """
        Run live localization from camera feed.

        Args:
            camera_id: Camera device ID (0 for built-in, 1+ for external)
            threshold: Recognition threshold
            show_top_k: Whether to show top-3 matches panel
        """
        print(f"\n{'='*60}")
        print(f"STARTING LIVE VPR TEST")
        print(f"{'='*60}")
        print(f"Camera ID: {camera_id}")
        print(f"Threshold: {threshold}")
        print(f"Database: {len(self.db_image_paths)} reference images")
        print(f"{'='*60}")
        print("\nControls:")
        print("  q     - Quit")
        print("  s     - Save current frame")
        print("  t     - Toggle top-k panel")
        print("  +/-   - Adjust threshold")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_id}")
            print("Try different camera IDs: 0, 1, 2...")
            return

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")

        frame_count = 0
        fps_start = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Localize
            result = self.localize(frame, threshold=threshold)

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()

            # Create display
            display = self._create_display(frame, result, threshold, fps, show_top_k)

            cv2.imshow('Live VPR - Press q to quit', display)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('t'):
                show_top_k = not show_top_k
            elif key == ord('+') or key == ord('='):
                threshold = min(1.0, threshold + 0.05)
                print(f"Threshold: {threshold:.2f}")
            elif key == ord('-'):
                threshold = max(0.0, threshold - 0.05)
                print(f"Threshold: {threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        print("\nLive test ended.")

    def _create_display(self, frame, result, threshold, fps, show_top_k):
        """Create the display frame with overlays."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Status bar background
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)

        # Recognition status
        if result['recognized']:
            color = (0, 255, 0)  # Green
            status = f"MATCH: #{result['best_match_idx']} (score: {result['best_score']:.3f})"
        else:
            color = (0, 0, 255)  # Red
            status = f"UNKNOWN (best: {result['best_score']:.3f})"

        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Info line
        info = f"Threshold: {threshold:.2f} | Latency: {result['extraction_time_ms']:.0f}ms | FPS: {fps:.1f}"
        cv2.putText(display, info, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show matched reference image
        if result['recognized']:
            try:
                ref_path = self.db_image_paths[result['best_match_idx']]
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    ref_h, ref_w = 150, 200
                    ref_img = cv2.resize(ref_img, (ref_w, ref_h))

                    # Add border
                    cv2.rectangle(display, (w-ref_w-15, 75), (w-5, 75+ref_h+10), color, 2)
                    display[80:80+ref_h, w-ref_w-10:w-10] = ref_img

                    # Label
                    cv2.putText(display, f"Ref #{result['best_match_idx']}",
                               (w-ref_w-10, 75+ref_h+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                pass

        # Show top-k matches panel
        if show_top_k and len(result['top_k_indices']) > 1:
            panel_y = h - 120
            cv2.rectangle(display, (0, panel_y-5), (w, h), (30, 30, 30), -1)
            cv2.putText(display, "Top matches:", (10, panel_y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            thumb_w, thumb_h = 100, 75
            for i, (idx, score) in enumerate(zip(result['top_k_indices'][:5],
                                                  result['top_k_scores'][:5])):
                x = 10 + i * (thumb_w + 10)
                if x + thumb_w > w - 10:
                    break

                try:
                    ref_img = cv2.imread(self.db_image_paths[idx])
                    if ref_img is not None:
                        ref_img = cv2.resize(ref_img, (thumb_w, thumb_h))
                        display[panel_y+25:panel_y+25+thumb_h, x:x+thumb_w] = ref_img

                        # Score label
                        score_color = (0, 255, 0) if score > threshold else (100, 100, 100)
                        cv2.putText(display, f"#{idx}: {score:.2f}", (x, panel_y+25+thumb_h+15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, score_color, 1)
                except:
                    pass

        return display

    def run_video_test(self, video_path, threshold=0.5, output_video=None):
        """
        Run localization on a recorded video.

        Args:
            video_path: Path to video file
            threshold: Recognition threshold
            output_video: Optional path to save annotated output video
        """
        print(f"\n{'='*60}")
        print(f"VIDEO TEST")
        print(f"{'='*60}")
        print(f"Input: {video_path}")
        print(f"Threshold: {threshold}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {total_frames} frames, {fps_video:.1f} FPS, {width}x{height}")
        print(f"{'='*60}\n")

        # Setup output video if requested
        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps_video, (width, height))

        results = []
        frame_idx = 0

        print("Processing... (press 'q' to stop early)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.localize(frame, threshold=threshold)
            result['frame_idx'] = frame_idx
            results.append(result)

            # Create display
            display = self._create_display(frame, result, threshold, 0, False)

            # Progress bar
            progress = int((frame_idx / total_frames) * 50)
            print(f"\r[{'='*progress}{' '*(50-progress)}] {frame_idx}/{total_frames}", end='')

            # Save to output video
            if out:
                out.write(display)

            # Show frame
            cv2.imshow('Video Test - Press q to stop', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped early by user")
                break

            frame_idx += 1

        print()  # New line after progress bar

        cap.release()
        if out:
            out.release()
            print(f"Output video saved to: {output_video}")
        cv2.destroyAllWindows()

        # Calculate statistics
        self._print_results_summary(results, threshold)

        return results

    def run_batch_eval(self, query_dir, threshold=0.5, gt_tolerance=5):
        """
        Run batch evaluation on a directory of query images.

        Args:
            query_dir: Directory containing query images
            threshold: Recognition threshold
            gt_tolerance: Frame index tolerance for ground truth matching

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION")
        print(f"{'='*60}")
        print(f"Query directory: {query_dir}")
        print(f"Threshold: {threshold}")
        print(f"GT tolerance: +/- {gt_tolerance} frames")
        print(f"{'='*60}\n")

        # Load query images
        query_dir = Path(query_dir)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        query_paths = []
        for ext in extensions:
            query_paths.extend(query_dir.glob(ext))
        query_paths = sorted([str(p) for p in query_paths])

        if len(query_paths) == 0:
            print(f"ERROR: No images found in {query_dir}")
            return None

        print(f"Found {len(query_paths)} query images")

        results = []
        for i, qpath in enumerate(query_paths):
            img = cv2.imread(qpath)
            if img is None:
                continue

            result = self.localize(img, threshold=threshold)
            result['query_path'] = qpath
            result['query_idx'] = i

            # Simple ground truth: assume query i should match reference ~i
            # (This works when both traversals have similar pacing)
            expected_idx = int(i * len(self.db_image_paths) / len(query_paths))
            result['expected_idx'] = expected_idx

            # Check if match is within tolerance
            if result['recognized']:
                match_idx = result['best_match_idx']
                result['correct'] = abs(match_idx - expected_idx) <= gt_tolerance
            else:
                result['correct'] = False

            results.append(result)

            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(query_paths)} queries")

        self._print_results_summary(results, threshold, show_accuracy=True)

        return results

    def _print_results_summary(self, results, threshold, show_accuracy=False):
        """Print summary statistics for test results."""
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")

        recognized = sum(1 for r in results if r['recognized'])
        total = len(results)

        scores = [r['best_score'] for r in results]
        latencies = [r['extraction_time_ms'] for r in results]

        print(f"Total frames/images: {total}")
        print(f"Recognized: {recognized} ({100*recognized/total:.1f}%)")
        print(f"Not recognized: {total-recognized} ({100*(total-recognized)/total:.1f}%)")
        print(f"\nSimilarity scores:")
        print(f"  Mean:   {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Min:    {np.min(scores):.3f}")
        print(f"  Max:    {np.max(scores):.3f}")
        print(f"\nLatency:")
        print(f"  Mean:   {np.mean(latencies):.1f}ms")
        print(f"  Median: {np.median(latencies):.1f}ms")

        if show_accuracy:
            correct = sum(1 for r in results if r.get('correct', False))
            print(f"\nAccuracy (within GT tolerance):")
            print(f"  Correct: {correct}/{total} ({100*correct/total:.1f}%)")

        print(f"{'='*60}\n")


def check_camera(camera_id=0):
    """Quick check if camera is accessible."""
    print(f"Testing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"FAILED: Cannot open camera {camera_id}")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"SUCCESS: Camera {camera_id} works!")
        print(f"Frame size: {frame.shape}")
        return True
    else:
        print(f"FAILED: Could not read frame from camera {camera_id}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Live VPR Testing for CMU-Africa Campus Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build database from day images
  python live_vpr_test.py --mode build_db --data_dir data/CMUAfrica/day --db_path day_db.npz

  # Live test with built-in camera
  python live_vpr_test.py --mode live_test --db_path day_db.npz --camera 0

  # Test with recorded video
  python live_vpr_test.py --mode video_test --db_path day_db.npz --video test.mp4

  # Batch evaluation
  python live_vpr_test.py --mode batch_eval --db_path day_db.npz --query_dir data/CMUAfrica/night

  # Check camera
  python live_vpr_test.py --mode check_camera --camera 0
        """
    )

    parser.add_argument('--mode',
                       choices=['build_db', 'live_test', 'video_test', 'batch_eval', 'check_camera'],
                       required=True,
                       help='Operation mode')
    parser.add_argument('--data_dir', type=str, default='data/CMUAfrica/day',
                       help='Directory with reference images (for build_db)')
    parser.add_argument('--db_path', type=str, default='database.npz',
                       help='Path to database file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (0=built-in, 1+=external)')
    parser.add_argument('--video', type=str,
                       help='Video path (for video_test)')
    parser.add_argument('--query_dir', type=str,
                       help='Query images directory (for batch_eval)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Recognition threshold (0.0-1.0)')
    parser.add_argument('--descriptor', type=str, default='CosPlace',
                       choices=['CosPlace', 'EigenPlaces', 'NetVLAD', 'HDC-DELF', 'AlexNet'],
                       help='Feature descriptor to use')
    parser.add_argument('--output_video', type=str,
                       help='Save annotated video to this path')
    parser.add_argument('--gt_tolerance', type=int, default=5,
                       help='Ground truth frame tolerance for batch_eval')

    args = parser.parse_args()

    # Quick camera check mode
    if args.mode == 'check_camera':
        check_camera(args.camera)
        return

    # Initialize VPR system
    vpr = LiveVPR(descriptor=args.descriptor)

    if args.mode == 'build_db':
        vpr.build_database(args.data_dir, args.db_path)

    elif args.mode == 'live_test':
        if not vpr.load_database(args.db_path):
            return
        vpr.run_live(camera_id=args.camera, threshold=args.threshold)

    elif args.mode == 'video_test':
        if not args.video:
            print("ERROR: --video required for video_test mode")
            return
        if not vpr.load_database(args.db_path):
            return
        vpr.run_video_test(args.video, threshold=args.threshold,
                          output_video=args.output_video)

    elif args.mode == 'batch_eval':
        if not args.query_dir:
            print("ERROR: --query_dir required for batch_eval mode")
            return
        if not vpr.load_database(args.db_path):
            return
        vpr.run_batch_eval(args.query_dir, threshold=args.threshold,
                          gt_tolerance=args.gt_tolerance)


if __name__ == '__main__':
    main()
