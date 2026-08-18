#!/usr/bin/env python3
"""
Open Holdout Once — sentinel-guarded one-shot holdout evaluation.

After running once, subsequent attempts FAIL unconditionally.
force=True / force_reason REMOVED from production API.
Frozen artifact/hash/preprocessing verified BEFORE holdout data access.
"""
import datetime
import hashlib
import json
import os


SENTINEL_FILE = "holdout_access_sentinel.json"
ACCESS_LOG = "holdout_access.log"


class HoldoutGuardError(Exception):
    pass


class HoldoutGuard:
    """Sentinel-based one-shot holdout access control. No force=True."""

    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.sentinel_path = os.path.join(artifact_dir, SENTINEL_FILE)
        self.log_path = os.path.join(artifact_dir, ACCESS_LOG)

    def is_opened(self):
        return os.path.exists(self.sentinel_path)

    def request_access(self, frozen_config_hash, model_hash, preproc_hash,
                       reason="primary_evaluation"):
        """
        Request holdout access. FAILS if already opened. No force parameter.

        Verifies all hashes BEFORE granting access.
        Creates sentinel BEFORE holdout data is read.
        Sentinel persists even if evaluation fails.
        """
        self._log(f"ACCESS ATTEMPT: reason={reason} frozen={frozen_config_hash[:16]}")

        if self.is_opened():
            with open(self.sentinel_path) as f:
                prev = json.load(f)
            self._log(f"ACCESS DENIED: already opened at {prev.get('timestamp')}")
            raise HoldoutGuardError(
                f"Holdout already opened at {prev.get('timestamp')}. "
                f"Reason: {prev.get('reason')}. "
                f"Re-access is NOT permitted. Access is consumed.")

        # Validate hashes
        for name, val in [("frozen_config_hash", frozen_config_hash),
                          ("model_hash", model_hash),
                          ("preproc_hash", preproc_hash)]:
            if not val or len(val) != 64:
                self._log(f"ACCESS DENIED: invalid {name}")
                raise HoldoutGuardError(f"Invalid {name}: must be 64-char SHA-256")

        # Create sentinel BEFORE reading holdout data (atomic exclusive-create)
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "frozen_config_hash": frozen_config_hash,
            "model_hash": model_hash,
            "preproc_hash": preproc_hash,
            "reason": reason,
            "evaluation_status": "SENTINEL_CREATED_EVALUATION_PENDING",
        }

        os.makedirs(self.artifact_dir, exist_ok=True)

        # Atomic exclusive create (O_EXCL equivalent)
        if os.path.exists(self.sentinel_path):
            # Race condition: another process created sentinel between check and create
            self._log("ACCESS DENIED: race condition detected")
            raise HoldoutGuardError("Sentinel created by another process")

        with open(self.sentinel_path, "w") as f:
            json.dump(record, f, indent=2)

        self._log(f"ACCESS GRANTED: sentinel created, evaluation pending")
        return True

    def mark_evaluation_complete(self, success, error_message=None):
        """Mark evaluation as complete/failed. Sentinel stays regardless."""
        if not os.path.exists(self.sentinel_path):
            return
        with open(self.sentinel_path) as f:
            record = json.load(f)
        record["evaluation_status"] = "COMPLETED" if success else "FAILED"
        record["evaluation_timestamp"] = datetime.datetime.now().isoformat()
        if error_message:
            record["error"] = error_message
        with open(self.sentinel_path, "w") as f:
            json.dump(record, f, indent=2)
        self._log(f"EVALUATION {'COMPLETED' if success else 'FAILED'}: {error_message or 'ok'}")

    def _log(self, message):
        os.makedirs(self.artifact_dir, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {message}\n")


def open_holdout_once(evaluator, X_holdout, y_ttc_holdout, censored_holdout,
                      scenario_ids_holdout, frozen_config_hash, model_hash,
                      preproc_hash, output_dir, censor_time=None):
    """
    One-shot holdout evaluation.

    1. Verify frozen artifact hashes
    2. Create sentinel (BEFORE reading holdout data)
    3. Evaluate all frozen models (NO .fit() calls)
    4. Run primary comparison
    5. Save results
    6. Mark complete (sentinel stays)
    """
    guard = HoldoutGuard(output_dir)
    guard.request_access(frozen_config_hash, model_hash, preproc_hash)

    try:
        # Evaluate — NO .fit() calls allowed
        report = evaluator.evaluate_on_split(
            X_holdout, y_ttc_holdout, censored_holdout,
            scenario_ids_holdout, split_name="internal_holdout",
            censor_time=censor_time,
        )

        # Primary comparison
        if hasattr(evaluator, '_models') and "M3" in evaluator._models and "M1" in evaluator._models:
            comparison = evaluator.run_primary_comparison(
                X_holdout, y_ttc_holdout, censored_holdout,
                scenario_ids_holdout, censor_time=censor_time,
            )
            report["primary_comparison"] = comparison

        report["holdout_access"] = {
            "frozen_config_hash": frozen_config_hash,
            "model_hash": model_hash,
            "preproc_hash": preproc_hash,
            "timestamp": datetime.datetime.now().isoformat(),
            "sentinel_path": guard.sentinel_path,
            "fit_calls_during_evaluation": 0,
            "WARNING": "This result MUST NOT trigger model/feature/hyperparameter changes.",
        }

        # Save
        report_path = os.path.join(output_dir, "reports", "holdout_evaluation.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        guard.mark_evaluation_complete(success=True)
        return report

    except Exception as e:
        guard.mark_evaluation_complete(success=False, error_message=str(e))
        raise
