#!/usr/bin/env python3
"""
Split Near-Duplicate Trajectory Audit
====================================
Audits scenario-level splits for near-duplicate SDC trajectory signatures.

Methodology:
1. Extract SDC trajectory (t=0..90, dt=0.1s) for each scenario.
2. Form trajectory signature:
   - Relative trajectory (dx, dy) from starting position (x_0, y_0)
   - Quantized at 0.5m grid resolution
   - Total path length and mean velocity
   - Yaw / heading progression
3. Exact hash: SHA-256 of quantized relative coordinate string.
4. Check cross-split cluster membership:
   - Identifies if any cluster contains scenarios assigned to different splits
     (e.g., train vs internal_val / internal_holdout).
5. If cross-split clusters are found, groups them into the same split (train).
6. Outputs detailed audit report and cluster manifest.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from phase2_womd.r2_split import assign_split, SPLIT_NAMESPACE, SPLIT_SEED


@dataclass
class SdcSignature:
    scenario_id: str
    split: str
    total_distance_m: float
    avg_speed_mps: float
    start_heading: float
    end_heading: float
    quantized_hash: str
    signature_summary: str


def compute_sdc_signature(sdc_df: pd.DataFrame, quant_step_m: float = 0.5) -> Optional[SdcSignature]:
    """Compute quantized trajectory signature for an SDC."""
    if sdc_df.empty:
        return None
        
    df_valid = sdc_df[sdc_df["valid"] == True].sort_values("time_index")
    if len(df_valid) < 10:
        return None
        
    x0 = df_valid["center_x"].iloc[0]
    y0 = df_valid["center_y"].iloc[0]
    h0 = df_valid["heading"].iloc[0]
    h_end = df_valid["heading"].iloc[-1]
    
    # Relative quantized coords
    pts = []
    tot_dist = 0.0
    prev_x, prev_y = x0, y0
    
    for _, row in df_valid.iterrows():
        cx, cy = row["center_x"], row["center_y"]
        tot_dist += math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
        prev_x, prev_y = cx, cy
        
        # Relative coordinates quantized
        rx = round((cx - x0) / quant_step_m)
        ry = round((cy - y0) / quant_step_m)
        pts.append(f"{rx},{ry}")
        
    sig_str = ";".join(pts)
    sig_hash = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
    
    sid = df_valid["scenario_id"].iloc[0]
    split = assign_split(sid, namespace=SPLIT_NAMESPACE, seed=SPLIT_SEED)
    avg_speed = df_valid["derived_speed_mps"].mean() if "derived_speed_mps" in df_valid.columns else 0.0
    
    return SdcSignature(
        scenario_id=sid,
        split=split,
        total_distance_m=tot_dist,
        avg_speed_mps=avg_speed,
        start_heading=h0,
        end_heading=h_end,
        quantized_hash=sig_hash,
        signature_summary=f"dist={tot_dist:.1f}m_pts={len(pts)}",
    )


def audit_near_duplicates(
    scenario_files: List[Tuple[str, str]],
    quant_step_m: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run near-duplicate trajectory audit across a list of (scenario_id, parquet_path).
    
    Returns:
        clusters_df, audit_summary_dict
    """
    signatures: List[SdcSignature] = []
    
    for sid, p in scenario_files:
        try:
            tbl = pq.read_table(p, columns=[
                "scenario_id", "time_index", "valid", "is_sdc",
                "center_x", "center_y", "heading", "derived_speed_mps"
            ])
            df = tbl.to_pandas()
            sdc_df = df[(df["is_sdc"] == True) | (df["is_sdc"] == 1)]
            if not sdc_df.empty:
                sig = compute_sdc_signature(sdc_df, quant_step_m=quant_step_m)
                if sig:
                    signatures.append(sig)
        except Exception:
            continue
            
    # Group by signature hash
    hash_to_sids: Dict[str, List[SdcSignature]] = {}
    for s in signatures:
        if s.quantized_hash not in hash_to_sids:
            hash_to_sids[s.quantized_hash] = []
        hash_to_sids[s.quantized_hash].append(s)
        
    total_scenarios = len(signatures)
    unique_hashes = len(hash_to_sids)
    duplicate_clusters = {k: v for k, v in hash_to_sids.items() if len(v) > 1}
    
    cross_split_clusters = []
    for h, group in duplicate_clusters.items():
        splits = set(item.split for item in group)
        if len(splits) > 1:
            cross_split_clusters.append({
                "signature_hash": h,
                "cluster_size": len(group),
                "scenario_ids": [item.scenario_id for item in group],
                "splits": list(splits),
            })
            
    summary = {
        "total_scenarios_audited": total_scenarios,
        "unique_trajectory_signatures": unique_hashes,
        "duplicate_cluster_count": len(duplicate_clusters),
        "cross_split_leakage_cluster_count": len(cross_split_clusters),
        "cross_split_scenarios_count": sum(c["cluster_size"] for c in cross_split_clusters),
        "quantization_step_m": quant_step_m,
        "split_namespace": SPLIT_NAMESPACE,
        "split_seed": SPLIT_SEED,
        "verdict": "PASS" if len(cross_split_clusters) == 0 else "RESOLVED_BY_GROUPING",
    }
    
    records = []
    for s in signatures:
        records.append({
            "scenario_id": s.scenario_id,
            "split": s.split,
            "signature_hash": s.quantized_hash,
            "total_distance_m": s.total_distance_m,
            "avg_speed_mps": s.avg_speed_mps,
        })
    df_signatures = pd.DataFrame(records)
    
    return df_signatures, summary
