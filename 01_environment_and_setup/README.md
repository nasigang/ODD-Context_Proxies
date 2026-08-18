# 01. Environment and Setup

This directory contains the complete environment definitions, Docker configuration, and dependency locks required to execute the entire pipeline.

## Files
- `Dockerfile`: Standalone container build with CUDA / CPU support, Python 3.10, and required system tools (tectonic, pdftoppm, pdfinfo).
- `compose.yaml`: Docker compose orchestration file.
- `docker.env`: Runtime environment variables.
- `phase2-core.txt`: Core scientific dependencies (numpy, pandas, pyarrow, scipy, shapely, matplotlib).
- `phase2-model.txt`: Machine learning dependencies (scikit-learn 1.7.1, joblib).
- `conda_environment_installed_packages.txt`: Full freeze of the active conda environment.
- `00_bootstrap_environment_for_antigravity.md`: Step-by-step bootstrap guide.
- `host_preflight.sh` / `verify_container.py`: Environment validation scripts.
