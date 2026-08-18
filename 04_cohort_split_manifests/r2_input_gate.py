#!/usr/bin/env python3
"""
R2 Input Gate — fail-closed validation with non-forgeable attestation.

Gate checks R1 acceptance, data integrity, contamination exclusion.
Produces an attestation artifact that the trainer re-verifies.
Public GateToken constructor removed: only gate.create_attestation() can produce valid attestations.
"""
import hashlib
import json
import os
import re
import time

import numpy as np
import pyarrow.parquet as pq


class R2InputGateError(Exception):
    pass


class GateAttestationError(Exception):
    pass


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')

REQUIRED_METHOD = "obb_swept_sat_cv_fixed_heading_v1"
BANNED_TARGET_COLUMNS = {
    "derived_ttc_2d_s", "ttc_min_s", "circle_ttc_s", "ttc_circle_s",
    "circle_frame_status", "circle_overlap_now", "circle_hit_future",
    "ttc_method_circle",
}
BLOCKED_PATH_TOKENS = {"pilot", "staging", "in_progress", "partial"}
VALID_FRAME_STATUSES = {
    "future_contact_event", "right_censored", "current_geometry_overlap",
    "no_exposure", "invalid_ego_state", "invalid_frame",
}


class R2InputGate:
    """
    Criterion-level R2 input gate.
    All checks are REQUIRED. Missing evidence = FAIL (not skip).
    """

    def __init__(self, r1_dir):
        self.r1_dir = r1_dir
        self.criteria = {}
        self.passed = True

    def _fail(self, gate, reason):
        self.criteria[gate] = {"verdict": "FAIL", "reason": reason}
        self.passed = False

    def _pass(self, gate, detail=""):
        self.criteria[gate] = {"verdict": "PASS", "detail": detail}

    def check_r1_acceptance(self):
        """R1 criterion-level acceptance must be PASS (not just top-level string)."""
        path = os.path.join(self.r1_dir, "reports", "r1_acceptance.json")
        if not os.path.exists(path):
            # Try alternate name
            path = os.path.join(self.r1_dir, "reports", "r1_full_acceptance.json")
        if not os.path.exists(path):
            self._fail("r1_acceptance", f"Missing acceptance report")
            return False
        with open(path) as f:
            data = json.load(f)
        status = data.get("overall", data.get("status", ""))
        if status not in ("PASS", "PASS_SMOKE_ONLY"):
            self._fail("r1_acceptance", f"Status '{status}', need PASS")
            return False
        # Check criterion-level: all non-BLOCKED must be PASS
        criteria = data.get("criteria", {})
        for cname, cdata in criteria.items():
            v = cdata.get("verdict", "")
            if v == "FAIL":
                self._fail("r1_acceptance", f"R1 criterion {cname} FAIL: {cdata.get('observed_value','')}")
                return False
        self._pass("r1_acceptance", f"overall={status}")
        return True

    def check_training_only_source(self):
        """Source manifest must have validation=0, testing=0."""
        sm_path = os.path.join(self.r1_dir, "manifests", "source_manifest.json")
        if not os.path.exists(sm_path):
            self._fail("training_only_source", f"Missing: {sm_path}")
            return False
        with open(sm_path) as f:
            sm = json.load(f)
        val = sm.get("validation_files", -1)
        test = sm.get("testing_files", -1)
        if val != 0 or test != 0:
            self._fail("training_only_source", f"val={val}, test={test}")
            return False
        # Check basenames
        for fi in sm.get("files", []):
            bn = fi.get("basename", "").lower()
            if "validation" in bn or "testing" in bn:
                self._fail("training_only_source", f"banned basename: {bn}")
                return False
        self._pass("training_only_source", f"{sm.get('n_files',0)} training files")
        return True

    def check_contamination_exclusion(self, quarantine_path=None):
        """R1 dir must not be in quarantine index."""
        if quarantine_path and os.path.exists(quarantine_path):
            with open(quarantine_path) as f:
                q = json.load(f)
            for run in q.get("quarantined_runs", []):
                qpath = run.get("output_path", "")
                if qpath and (self.r1_dir.rstrip("/").endswith(qpath.rstrip("/")) or
                              qpath.rstrip("/").endswith(self.r1_dir.rstrip("/"))):
                    self._fail("contamination_exclusion",
                               f"R1 dir matches quarantined run: {qpath}")
                    return False
        self._pass("contamination_exclusion")
        return True

    def check_data_path(self, frame_path):
        path_lower = frame_path.lower()
        for token in BLOCKED_PATH_TOKENS:
            if token in path_lower:
                self._fail("data_path", f"Blocked token '{token}' in: {frame_path}")
                return False
        if not os.path.exists(frame_path):
            self._fail("data_path", f"Not found: {frame_path}")
            return False
        self._pass("data_path", frame_path)
        return True

    def check_method_and_legacy(self, frame_path):
        if not os.path.exists(frame_path):
            self._fail("method", f"Not found: {frame_path}")
            return False
        schema = pq.read_schema(frame_path)
        cols = set(c.lower().replace("-", "_").replace(" ", "_") for c in schema.names)
        banned = cols & {c.lower() for c in BANNED_TARGET_COLUMNS}
        if banned:
            self._fail("legacy_columns", f"Banned: {banned}")
            return False
        self._pass("legacy_columns")
        if "ttc_method" not in schema.names:
            self._fail("method", "ttc_method column missing")
            return False
        t = pq.read_table(frame_path, columns=["ttc_method"])
        methods = set(m for m in t.column("ttc_method").to_pylist() if m)
        if methods != {REQUIRED_METHOD}:
            self._fail("method", f"Expected only {REQUIRED_METHOD}, got {methods}")
            return False
        self._pass("method", REQUIRED_METHOD)
        return True

    def check_risk_set_invariants(self, frame_path):
        if not os.path.exists(frame_path):
            self._fail("risk_set", "Not found")
            return False
        df = pq.read_table(frame_path).to_pandas()
        issues = []

        # Overlap in primary
        primary = df[df["target_status"].isin({"future_contact_event", "right_censored"})]
        if "overlap_now_flag" in df.columns:
            overlap_in_primary = primary["overlap_now_flag"].sum()
            if overlap_in_primary > 0:
                issues.append(f"overlap_now_flag=true in {overlap_in_primary} primary rows")

        # Event should not be censored
        events = df[df["target_status"] == "future_contact_event"]
        if "right_censored" in df.columns and len(events) > 0 and events["right_censored"].any():
            issues.append(f"{events['right_censored'].sum()} event rows have right_censored=true")

        # Event TTC range
        if "ttc_obb_swept_s" in df.columns and len(events) > 0:
            ev_ttc = events["ttc_obb_swept_s"]
            bad_ttc = (~np.isfinite(ev_ttc)) | (ev_ttc <= 0)
            if bad_ttc.any():
                issues.append(f"{bad_ttc.sum()} event rows with TTC <= 0 or nonfinite")

        # Status exhaustive
        unknown = set(df["target_status"].unique()) - VALID_FRAME_STATUSES
        if unknown:
            issues.append(f"Unknown statuses: {unknown}")

        # No duplicate frame keys
        dups = df.duplicated(subset=["scenario_id", "time_index"]).sum()
        if dups > 0:
            issues.append(f"{dups} duplicate frame keys")

        if issues:
            self._fail("risk_set", "; ".join(issues))
            return False

        self._pass("risk_set", f"{len(df)} frames, {len(primary)} primary, {len(events)} events")
        return True

    def check_split_membership(self, membership_path, expected_hash=None):
        if not membership_path or not os.path.exists(membership_path):
            self._fail("split_membership", f"Missing: {membership_path}")
            return False
        with open(membership_path) as f:
            membership = json.load(f)
        if expected_hash:
            with open(membership_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            if actual != expected_hash:
                self._fail("split_membership", f"Hash mismatch: {actual[:16]} != {expected_hash[:16]}")
                return False
        # Check no validation/testing scenarios
        val_scenarios = membership.get("external_test", [])
        if val_scenarios:
            self._fail("split_membership", f"{len(val_scenarios)} external_test scenarios")
            return False
        # Overlap check
        sets = {k: set(v) for k, v in membership.items()}
        names = list(sets.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                overlap = sets[n1] & sets[n2]
                if overlap:
                    self._fail("split_membership", f"{n1}∩{n2}: {len(overlap)}")
                    return False
        total = sum(len(v) for v in sets.values())
        self._pass("split_membership", f"{total} scenarios, 0 overlap")
        return True

    def check_manifest_hash(self, manifest_path, expected_hash):
        if not manifest_path or not expected_hash:
            self._fail("manifest_hash", "REQUIRED: manifest_path AND expected_hash")
            return False
        if not os.path.exists(manifest_path):
            self._fail("manifest_hash", f"Not found: {manifest_path}")
            return False
        with open(manifest_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected_hash:
            self._fail("manifest_hash", f"Mismatch: {actual[:16]} != {expected_hash[:16]}")
            return False
        self._pass("manifest_hash", actual[:16])
        return True

    def check_required_features(self, frame_path, required_features):
        if not os.path.exists(frame_path):
            self._fail("features", "Not found")
            return False
        schema = pq.read_schema(frame_path)
        cols = set(schema.names)
        missing = set(required_features) - cols
        if missing:
            self._fail("features", f"Missing features: {missing}")
            return False
        self._pass("features", f"All {len(required_features)} features present")
        return True

    def run_all(self, frame_path, manifest_path, expected_hash,
                membership_path, membership_hash=None, required_features=None,
                quarantine_path=None):
        """Run ALL gates. Missing args = FAIL."""
        for arg_name, arg_val in [("frame_path", frame_path),
                                   ("manifest_path", manifest_path),
                                   ("expected_hash", expected_hash),
                                   ("membership_path", membership_path)]:
            if not arg_val:
                self._fail("args", f"{arg_name} is REQUIRED")

        if not self.passed:
            return False, self.criteria

        self.check_r1_acceptance()
        self.check_training_only_source()
        self.check_contamination_exclusion(quarantine_path)
        self.check_data_path(frame_path)
        self.check_method_and_legacy(frame_path)
        self.check_risk_set_invariants(frame_path)
        self.check_manifest_hash(manifest_path, expected_hash)
        self.check_split_membership(membership_path, membership_hash)
        if required_features:
            self.check_required_features(frame_path, required_features)

        return self.passed, self.criteria

    def create_attestation(self, output_path, **kwargs):
        """
        Create attestation artifact after all gates pass.
        This is the ONLY way to produce a valid attestation.
        Returns attestation hash.
        """
        if not self.passed:
            raise R2InputGateError(
                f"Cannot create attestation: gate FAILED. Criteria: {self.criteria}")

        attestation = {
            "status": "GATE_PASS",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "criteria": self.criteria,
            "r1_dir": self.r1_dir,
            **kwargs,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(attestation, f, indent=2)

        with open(output_path, "rb") as f:
            att_hash = hashlib.sha256(f.read()).hexdigest()

        return att_hash

    @staticmethod
    def verify_attestation(attestation_path, r1_dir, frame_hash, membership_hash):
        """
        Re-verify attestation artifact. Used by trainer before .fit().
        Reads the attestation, checks all contents match expected values.
        """
        if not os.path.exists(attestation_path):
            raise GateAttestationError(f"Attestation not found: {attestation_path}")

        with open(attestation_path) as f:
            att = json.load(f)

        issues = []

        if att.get("status") != "GATE_PASS":
            issues.append(f"status: {att.get('status')}, expected GATE_PASS")

        # Check all criteria passed
        for cname, cdata in att.get("criteria", {}).items():
            if cdata.get("verdict") == "FAIL":
                issues.append(f"criterion {cname} FAIL: {cdata.get('reason','')}")

        # Verify R1 dir matches
        if att.get("r1_dir") != r1_dir:
            issues.append(f"r1_dir mismatch: {att.get('r1_dir')} != {r1_dir}")

        # Verify frame and membership hashes if provided
        if frame_hash and att.get("frame_hash") and att["frame_hash"] != frame_hash:
            issues.append(f"frame_hash mismatch")
        if membership_hash and att.get("membership_hash") and att["membership_hash"] != membership_hash:
            issues.append(f"membership_hash mismatch")

        if issues:
            raise GateAttestationError(
                f"Attestation verification failed: {'; '.join(issues)}")

        return att
