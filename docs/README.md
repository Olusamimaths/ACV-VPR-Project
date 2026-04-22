# Documentation Guide

This folder now has a few documents with different purposes. If you are new to the project, use this page to decide where to start.

## Start Here

If you want the best first overview, read:

1. [PROJECT_ARCHITECTURE_GUIDE.md](./architecture/PROJECT_ARCHITECTURE_GUIDE.md)
2. [LIVE_VPR_PIPELINE.md](./live-vpr/LIVE_VPR_PIPELINE.md)

That gives you:

- the overall repo structure
- the difference between the benchmark path and the live path
- the main files and modules to read next

## Which Doc Should I Open?

### I want to understand how the codebase works

Open:

- [PROJECT_ARCHITECTURE_GUIDE.md](./architecture/PROJECT_ARCHITECTURE_GUIDE.md)

Use this for:

- onboarding
- file/module ownership
- understanding the benchmark and live architectures
- finding where to implement changes

### I want to understand the live pipeline specifically

Open:

- [LIVE_VPR_PIPELINE.md](./live-vpr/LIVE_VPR_PIPELINE.md)

Use this for:

- offline map building
- traversal recording
- online localization
- live UI and saved inference reports
- source aliases and stream handling

### I want short commands I can copy and run

Open:

- [LIVE_VPR_COMMANDS.md](./live-vpr/LIVE_VPR_COMMANDS.md)

Use this for:

- bash commands
- raw Python commands
- webcam and phone-webcam flows
- TurboPi stream commands

### I want to use the bash launcher

Open:

- [LIVE_VPR_SCRIPT.md](./live-vpr/LIVE_VPR_SCRIPT.md)

Use this for:

- `scripts/live_vpr_cli.sh`
- launcher commands
- environment-variable defaults
- understanding what the launcher wraps

### I want to work with the campus dataset

Open:

- [CAMPUS_USER_GUIDE.md](./live-vpr/CAMPUS_USER_GUIDE.md)

Use this for:

- campus benchmark evaluation
- dataset structure and ground truth
- campus live-map workflows
- where campus-specific logic lives

### I want the metric definitions

Open:

- [VPR_EVALUATION_FLOW.md](./architecture/VPR_EVALUATION_FLOW.md)

Use this for:

- similarity matrix interpretation
- matching behavior
- `AUC`
- `R@100P`
- `R@K`

## Suggested Reading Paths

### New Developer

1. [PROJECT_ARCHITECTURE_GUIDE.md](./architecture/PROJECT_ARCHITECTURE_GUIDE.md)
2. [LIVE_VPR_PIPELINE.md](./live-vpr/LIVE_VPR_PIPELINE.md)
3. [LIVE_VPR_COMMANDS.md](./live-vpr/LIVE_VPR_COMMANDS.md)

### Demo / Presentation Prep

1. [LIVE_VPR_COMMANDS.md](./live-vpr/LIVE_VPR_COMMANDS.md)
2. [LIVE_VPR_SCRIPT.md](./live-vpr/LIVE_VPR_SCRIPT.md)
3. [CAMPUS_USER_GUIDE.md](./live-vpr/CAMPUS_USER_GUIDE.md)

### Research / Evaluation Work

1. [PROJECT_ARCHITECTURE_GUIDE.md](./architecture/PROJECT_ARCHITECTURE_GUIDE.md)
2. [VPR_EVALUATION_FLOW.md](./architecture/VPR_EVALUATION_FLOW.md)
3. [CAMPUS_USER_GUIDE.md](./live-vpr/CAMPUS_USER_GUIDE.md)

## One-Sentence Summary Of Each Doc

- [PROJECT_ARCHITECTURE_GUIDE.md](./architecture/PROJECT_ARCHITECTURE_GUIDE.md): the main onboarding and codebase map
- [LIVE_VPR_PIPELINE.md](./live-vpr/LIVE_VPR_PIPELINE.md): how the modular live system works
- [LIVE_VPR_SCRIPT.md](./live-vpr/LIVE_VPR_SCRIPT.md): how to use the bash wrapper
- [LIVE_VPR_COMMANDS.md](./live-vpr/LIVE_VPR_COMMANDS.md): copy-paste command cookbook
- [CAMPUS_USER_GUIDE.md](./live-vpr/CAMPUS_USER_GUIDE.md): how to use the campus dataset in evaluation and live workflows
- [VPR_EVALUATION_FLOW.md](./architecture/VPR_EVALUATION_FLOW.md): metric and evaluation logic reference
