# Completed BO research and independent validation — 2026-10-26

This branch archives the completed BO study. It is **not** a new accepted plant,
not a held-out PASS, and does not contain new RL, shadow or gate implementation.
The immutable plant source is `4ec49538912fe88ca3be569c1a787dba4c07994b` and the
accepted Docker image is
`sha256:c7dd2ca9507f72a6937b694f7318560a1f6c923035e561a9884d2b68458c16bc`.
The image was built from identical frozen six-file contents before that source
commit was created. This archival commit does not rebuild the image.

## Results

- Frozen `bo_campaign_contract_v1_patch1`, Framework v2, fixed evaluation no-go
  reference 60 degrees, framework/runtime-invalid costs 5/10.
- Six original BO dimensions/ranges; seed 42; 2 repeats x 3 Core scenarios.
- 40 iterations / 240 evaluated slots; 30/40 feasible candidates,
  223/240 feasible trials, 239/240 runtime-valid trials.
- Incumbent iteration 14: objective **0.32607422077940834**,
  17.52% below fresh campaign iteration 0. This is a search result, not an
  independent generalization estimate.
- Independent fresh Core-3: A and B each 6/6 feasible. Mean effective costs
  **A 0.3918545887115951**, **B 0.34561837573845466**: **11.80% improvement**.
- Held-out Long-Oblique r00: both A and B failed `excessive_roll`, with maxima
  52.29 and 53.17 degrees. Runtime/L1 PASS, L2 BOUNDED TRANSIENT, L3 FAIL,
  effective cost 5 each. Fail-fast stopped after **14/20** trials; the other six
  were not executed. The original STOP is preserved, not waived or deleted.
- B is a **provisional optimized/Core-3 validated baseline**, not a generalized
  or production-safe controller. No statistical significance is claimed from
  two repeats. Common roll failure and known pitch/daggerboard limitations are
  unresolved; physics debugging is deferred.

`params_iter014.json` is the exact flat controller input, SHA256
`1e52aed28be6c660e55fe831cb718411dd38edee3b66c670db0125d18a4fe663`.
Do not substitute the old `verified_incumbent_params.json`. The reference is
the exact iter_008 JSON used as fresh campaign iteration 0.

## What GitHub does and does not preserve

Git stores the used optimizer/evaluators/contracts/scenarios/tests and independent
validation tooling, exact A/B inputs, campaign metadata/history, compact 40- and
240-row results, 14 validation records, the original STOP snapshot and SHA
manifest. Original raw files stay in their existing local locations; they are
**not uploaded**. The immutable Docker image ID is recorded, but Docker image
layers are **not stored in GitHub**. Preserve/export the image separately if
independent image recovery is required.

`manifest.json` records code/config/source hashes, source/image lineage, original
workspace state, result locations and limitations. `raw_inventory.jsonl` covers
the campaign plus external iteration-0, iteration-1 recovery dependencies and
accepted-parent validation records. Correction history remains visible:

- Original v1 iteration-0 evaluation is retained; only corrected patch1 evidence
  seeded this fresh BO history.
- BO iteration1 Crosswind r01 OPEN and penalty5 remain unchanged. Recovery reused
  completed Crosswind trials and the saved proposal, ran only missing slots and
  appended one observation; the OPEN was not reclassified as a PASS.
- Separately, accepted-parent Core-3 r01 Crosswind had completed normally before
  orchestration stopped on missing evaluator functions. Its frozen-v2 offline
  recovery gate PASS is recorded with the original accepted-parent evidence;
  that orchestration issue was not a physics/navigation failure.
- Iteration34 runtime-invalid and cost10 remain committed in the 40 observations.

## Read-only verification and local raw backup

Run from this repository on WSL; these commands never launch a simulation:

```bash
python3 -B research/bo_iter014_20261026/preserve.py verify
```

To make a **fresh separate copy**, first check `df -h` and the logical byte size
in the manifest. Originals are never moved or deleted:

```bash
python3 -B research/bo_iter014_20261026/preserve.py backup \
  --destination "$HOME/sailboat_research_backups/bo_iter014_20261026"
```

The helper refuses an existing destination, checks free space, copies and verifies
every recorded file. It creates BACKUP_VERIFIED.json only after SHA verification.
This location is on the same WSL disk: it protects against accidental source-file
changes, **not disk/WSL loss**. Copy that verified mirror and the immutable Docker
image to independent storage for disaster recovery. To verify a relocated mirror:

```bash
python3 -B research/bo_iter014_20261026/preserve.py verify \
  --results-root /path/to/independent/mirror
```

Frozen evaluation artifacts contain original absolute paths. Restore the original
layout before using production preflight, or use the mirror verifier above;
never edit frozen artifacts to make paths look current.

## Tests and reproduction scope

Before the archival commit: BO unit/fixtures 80 PASS, validation 23 PASS,
Framework-v2 8 PASS, preservation-helper fixtures 6 PASS; no actual Gazebo/BO run. Some provenance integration tests
require the original local inputs and accepted source HEAD; they are not generic
asset-free CI tests. Frozen validation `prepare` intentionally requires accepted
HEAD 4ec4953, not this archival documentation commit. For a fresh reproduction,
use a clean accepted-source worktree plus the byte-identical BO tooling overlay,
pinned image and restored input artifacts. Never resume the completed campaign
to regenerate a historical observation.

RL work belongs on a separate branch/workspace. No RL edits should be added to
this archival branch.
