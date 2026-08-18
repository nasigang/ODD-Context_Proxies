# 02. Core Geometry, Kinematics and WOMD Parsing

This directory contains the foundational geometric algorithms, kinematic equations, and Waymo Open Motion Dataset (WOMD) TFRecord parsers.

## Core Modules
- `obb_ttc.py`: Oriented Bounding Box (OBB) representation, Separating Axis Theorem (SAT) collision detection, and instantaneous Time-to-Collision (TTC) calculations.
- `obb_ttc_swept.py`: Continuous swept SAT OBB-TTC solver over dynamic multi-actor motion trajectories.
- `kinematics.py`: Coordinate frame transformations, longitudinal/lateral relative velocities, path projections, and clearances.
- `fast_ttc.py`: Vectorized Euclidean and path-relative TTC helper utilities.
- `parser.py`: High-throughput TFRecord parser extracting vehicle/pedestrian/cyclist tracks, SDC ego state, static road graphs, and dynamic traffic signals.
- `map_features.py` & `signal_features.py`: Road geometry polylines, crosswalk polygons, stop signs, and traffic light phase associations.
- `schema.py`: Canonical schema definitions for frames, scenarios, and feature records.
- `womd_identity_audit.py`: Track ID continuity and ego identity auditor.
