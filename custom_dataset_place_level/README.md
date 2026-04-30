# Campus Dataset Place-Level Rematch

This folder is a **duplicate** of `custom_dataset/` created after a manual visual review of the day/night image pairs.

The original dataset is still preserved unchanged.

## Why This Duplicate Exists

The original campus evaluation assumes mostly **1-to-1 image matching**:

- one night query
- one exact daytime reference image

After reviewing the image pairs visually, many of the so-called "false matches" appear to be:

- the **correct physical place**
- but from a **different acceptable daytime viewpoint**

So this duplicate dataset reframes the campus benchmark at the **place level** rather than the strict image level.

## What Changed

The image files themselves are copied as-is, but this folder adds manual annotations:

- `place_flow.csv`
  - the ordered flow of places along the campus traversal
- `day_place_index.csv`
  - the place assignment for each day/reference image
- `night_rematches.csv`
  - the rematched place-level assignment for each night/query image

## How To Read The Annotations

### `place_flow.csv`

Defines the main sequence of places in the campus walk.

### `day_place_index.csv`

Maps each database image to a broader place ID.

### `night_rematches.csv`

Maps each night query to:

- its original label status
- the rematched place ID
- the set of valid daytime reference images for that place
- a short note explaining the rematch

## Important Interpretation

This is still a **manual visual rematch**, not a surveyed geometric ground truth.

It should be treated as:

- a better approximation of **place-level localization**
- useful for re-evaluating the current descriptors more fairly
- especially useful for understanding same-place / different-view retrieval

## Suggested Next Step

If you later want to evaluate this duplicate dataset in code, the cleanest change would be:

- load the `night_rematches.csv`
- treat `valid_day_images` as the positive set for each night query
- compute both:
  - strict image-level metrics
  - place-level multi-positive metrics
