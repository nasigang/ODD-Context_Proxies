# 04. Cohort Partitioning and Split Manifests

Defines the authoritative scenario-disjoint partitioning of the 18,445 WOMD scenarios into development and holdout cohorts.

## Partitioning Summary
- **Total Scenarios**: 18,445 ($1,677,495$ frames)
- **Development Cohort (84.8%)**: 15,641 scenarios ($1,422,331$ frames)
  - Training Split: 10,948 scenarios
  - Tuning Split: 4,693 scenarios
- **Sealed Holdout Cohort (15.2%)**: 2,804 scenarios ($255,164$ frames)
  - Accessed exactly once in Phase 2E under locked protocol.
- **Audit**: `split_near_duplicate_audit.py` guarantees 0 scenario-level or trajectory-level cross-split contamination.
