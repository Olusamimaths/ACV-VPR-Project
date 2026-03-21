Data Collection and Preprocessing Report
Rhoda Ojetola Peter Adeyemo Samuel Olusola
March 2026
1 Data Collection
To complement the GardensPoint benchmark and evaluate our methods on a local environment,
we collected a custom dataset on the Carnegie Mellon University Africa campus in Kigali, Rwanda.
1.1 Collection Protocol
A single outdoor path was traversed twice: once during daylight (approximately 15:00–16:00 local
time) and once after sunset (approximately 18:00–18:30). Both still images and video were cap-
tured using a smartphone camera (Google Pixel) held at chest height to simulate a forward-facing
robot camera. The daytime and nighttime traversals followed the same route to ensure spatial
correspondence between the two conditions.
1.2 Raw Data
The collection yielded:
• Still images: 114 photographs (50 day, 64 night) at native resolution (8160×4590 pixels, ∼6 MB
each).
• Video: 10 clips totalling approximately 2 minutes (4 day, 6 night) at 1920×1080 resolution.
1.3 Preprocessing
All images were processed to match the GardensPoint format:
1. Resolution normalisation: Images and video frames were resized to 640×480 pixels using
FFmpeg, consistent with the GardensPoint benchmark.
2. Frame extraction: Video clips were sampled at 2 FPS to avoid redundant frames while pre-
serving sequence continuity.
3. Day/night separation: Files were automatically sorted into day and night subsets based on
capture timestamps embedded in the filenames.
The preprocessing reduced file sizes by approximately 99% (from ∼6 MB to ∼65 KB per image),
enabling efficient descriptor extraction and similarity computation.
1
1.4 Final Dataset
Table 1: CMU-Africa campus dataset after preprocessing.
Source Day Night
Still images 50 64
Video frames (2 FPS) 82 159
Total frames 132 223
This dataset provides a locally relevant testbed with similar characteristics to GardensPoint: se-
quential traversal, severe day-to-night illumination shift, and no structural changes between condi-
tions.