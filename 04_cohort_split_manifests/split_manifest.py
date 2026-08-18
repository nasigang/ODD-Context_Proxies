#!/usr/bin/env python3
"""
Scenario Split Manifest Generator
=================================
Generates deterministic scenario-level split manifest for all 18,445 scenarios
using canonical SHA-256 hash (namespace: womd_r2_split_v1, seed: 42).

Split Ratios:
- train: 70%
- internal_val: 15%
- internal_holdout: 15% (SEALED)
"""

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import pandas as pd

from phase2_womd.r2_split import (
    assign_split,
    deterministic_split_hash,
    SPLIT_NAMESPACE,
    SPLIT_SEED,
)


def generate_split_manifest(
    parquet_root: str,
    output_csv: str = "work/SPLIT_MANIFEST.csv",
) -> pd.DataFrame:
    """Generate exact split manifest for all discovered scenario partitions."""
    agent_dir = os.path.join(parquet_root, "agent_state")
    if not os.path.isdir(agent_dir):
        agent_dir = parquet_root
        
    pattern = os.path.join(agent_dir, "scenario_id=*", "*.parquet")
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(agent_dir, "*.parquet"))
        
    scenario_ids = set()
    for f in files:
        parent = os.path.basename(os.path.dirname(f))
        if parent.startswith("scenario_id="):
            sid = parent.split("=")[1]
        else:
            sid = os.path.splitext(os.path.basename(f))[0]
        scenario_ids.add(sid)
        
    sorted_sids = sorted(scenario_ids)
    records = []
    for sid in sorted_sids:
        u = deterministic_split_hash(SPLIT_NAMESPACE, SPLIT_SEED, sid)
        split = assign_split(sid, SPLIT_NAMESPACE, SPLIT_SEED)
        records.append({
            "scenario_id": sid,
            "split": split,
            "hash_float": round(u, 8),
        })
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Generated split manifest for {len(df)} scenarios -> {output_csv}")
    print("Split distribution:")
    print(df["split"].value_counts(normalize=True))
    print(df["split"].value_counts())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_root", type=str, default="/home/kiapi/waymo_motion_project/runtime/outputs/model/parquet")
    parser.add_argument("--output_csv", type=str, default="work/SPLIT_MANIFEST.csv")
    args = parser.parse_args()
    generate_split_manifest(args.parquet_root, args.output_csv)
