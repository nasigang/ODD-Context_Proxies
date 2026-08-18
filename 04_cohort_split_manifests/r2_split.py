#!/usr/bin/env python3
"""
R2 Split — deterministic hash-based scenario-level split.

Generated from full scenario manifest ONCE, frozen.
Same scenario always gets same split regardless of row order, pilot/full, re-execution.
"""
import hashlib
import json
import os
import struct

import numpy as np


SPLIT_NAMESPACE = "womd_r2_split_v1"
SPLIT_SEED = 42
SPLIT_THRESHOLDS = {"train": 0.70, "internal_val": 0.85}  # 70/15/15


def deterministic_split_hash(namespace, seed, scenario_id):
    """
    Map scenario_id to [0,1) deterministically.

    u = hash(namespace || seed || scenario_id) → float in [0,1)
    """
    key = f"{namespace}|{seed}|{scenario_id}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    int_val = struct.unpack(">Q", h[:8])[0]
    return int_val / (2**64)


def assign_split(scenario_id, namespace=SPLIT_NAMESPACE, seed=SPLIT_SEED):
    """Assign a single scenario to its split."""
    u = deterministic_split_hash(namespace, seed, scenario_id)
    if u < SPLIT_THRESHOLDS["train"]:
        return "train"
    elif u < SPLIT_THRESHOLDS["internal_val"]:
        return "internal_val"
    else:
        return "internal_holdout"


def generate_split_membership(scenario_ids, namespace=SPLIT_NAMESPACE, seed=SPLIT_SEED):
    """
    Generate exact split membership from scenario list.

    Deterministic, order-invariant, pilot/full consistent.
    """
    membership = {"train": [], "internal_val": [], "internal_holdout": []}
    for sid in sorted(set(scenario_ids)):
        split = assign_split(sid, namespace, seed)
        membership[split].append(sid)

    # Verify
    all_assigned = set()
    for split, sids in membership.items():
        s = set(sids)
        assert len(s) == len(sids), f"Duplicate in {split}"
        assert len(all_assigned & s) == 0, f"Overlap with {split}"
        all_assigned |= s

    assert all_assigned == set(scenario_ids), "Not all scenarios assigned"

    return membership


def save_frozen_membership(membership, output_dir):
    """Save membership with hash. Used as frozen reference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "exact_scenario_split_membership.json")
    with open(path, "w") as f:
        json.dump(membership, f, indent=2)
    with open(path, "rb") as f:
        mem_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = os.path.join(output_dir, "exact_scenario_split_membership_hash.json")
    with open(hash_path, "w") as f:
        json.dump({
            "hash": mem_hash,
            "namespace": SPLIT_NAMESPACE,
            "seed": SPLIT_SEED,
            "n_scenarios": sum(len(v) for v in membership.values()),
            "n_train": len(membership["train"]),
            "n_internal_val": len(membership["internal_val"]),
            "n_internal_holdout": len(membership["internal_holdout"]),
        }, f, indent=2)
    return path, mem_hash


def load_frozen_membership(path):
    """Load frozen membership. Do NOT regenerate."""
    with open(path) as f:
        return json.load(f)
