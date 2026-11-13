# CS224W FIMS milestone: source and result companion

This is a source snapshot and evidence companion to the milestone report.
It is not a finished Colab, a standalone portable training tutorial, or a
complete data/model release. Opening this ZIP does not launch any cluster jobs.

## Contents

- `radius_source/`: the exact frozen radius-sensitivity source package,
  including the original PyTorch Geometric implementation and memory-recovery
  implementation under `vendor/recovery/`. Its 67 manifest members are preserved
  byte for byte. This includes dataset loading, graph construction, model
  definitions, training, deterministic controls, metrics, reporting, tests,
  experiment design, and cluster/local entry points.
- `operating_point_source/`: the exact frozen CPU reporting package for the
  22-fit operating-point review, including its required vendored source and
  frozen reference metadata. Its 35 manifest members are preserved byte for byte.
- `reported_results/`: the supplied operating-point CSV, metrics JSON,
  provenance JSON, summary, and original report completion manifest.
- `COMPANION_VERIFICATION.json`: checks performed while packaging.
- `SHA256SUMS.json`: hashes of every other file included in this companion.

The original report completion manifest describes the full upstream report,
including files not included here. It must not be interpreted as a claim that
all upstream predictions, arrays, plots, and per-event rows are bundled.
Use SHA256SUMS.json for the contents of this ZIP.

## Scientific scope and completed work

The reported development comparison comprises 22 completed fits:

1. Four GNN configurations (one layer, two layers, two layers without explicit
   x/y offsets, and two layers using only the current bin), each with three
   training seeds: 12 GNN fits.
2. Small and wider CNN references, each with three seeds: six CNN fits.
3. Counts-only and counts-plus-shape controls, each using either the current
   bin or eight bins: four deterministic control fits.

The seeds are 20260924, 20260925, and 20260926. Deterministic controls have one
fit and should not be presented as three independent training repetitions.
The development sample has 7,500 training and 2,500 validation events. Data
partitions are by event. The task is binary site classification with 25 ns
time bins, not event classification or continuous-arrival classification.

The operating-point CSV contains 110 rows (22 fits x five retention targets):
99.9%, 99.5%, 99%, 95%, and 90%. Cutoffs were selected and reported on the same
development-validation events. These are developmental results, not independent
confirmation. The CSV records per-fit results; neural three-seed means can be
computed from those rows without rerunning inference.

Primary-charge attribution and truth ancestry are reporting/label information,
not observable model inputs. Ignored targets are excluded from supervised
classification metrics but charge at flagged ignored sites still counts in
the primary-charge loss. Event-damage fractions use primary-bearing events.

The existing simulation, truth labels, and earlier non-graph studies are prior
FIMS research infrastructure. The new CS224W contributions include graph
construction, GNN comparisons and ablations, matched feature controls, the
radius study, and the planned tutorial/reproducible release.

## Radius sensitivity: status at the milestone snapshot

The radius study changes the neighborhood radius to 1, 2, or 4 mm, using a
fresh one-layer GNN with three seeds at each radius. This is nine additional
GNN fits, separate from the 22 completed fits above. Six matched deterministic
controls (counts and counts plus shape at each radius) were also submitted.

In the latest supplied scheduler snapshot, all nine radius GNN fits were
pending and had not started. All six control completion markers were present;
the 4 mm control allocation timed out after printing completion messages for
both fits. Full saved-output verification is still required before those
control results are treated as verified. No radius fit outputs or unverified
control metrics are included in this ZIP. The radius source package is evidence
of the implemented experimental protocol, not evidence of completed radius GNN
training or a particular radius outperforming another.

## Source map for a reviewer

- Core PyG graph/model: `radius_source/vendor/recovery/vendor/original/code/graph_models_v1.py`
- Dataset interface: `radius_source/vendor/recovery/vendor/original/code/site_data_v1.py`
- Original training: `radius_source/vendor/recovery/vendor/original/code/train_site_gnn_v1.py`
- Controls: `radius_source/vendor/recovery/vendor/original/code/fit_controls_v1.py`
- Memory-bounded execution: `radius_source/vendor/recovery/code/memory_model.py`
- Radius construction/training: `radius_source/code/radius_graph.py` and `train_radius.py`
- Radius controls/report: `radius_source/code/control_radius.py` and `report_radius.py`
- Operating-point report: `operating_point_source/code/report_operating_points.py`

The source test files contain synthetic test-case construction. They are
implementation checks, not scientific results. No generated synthetic fixtures
or test-output directories are included.

## Execution and remaining dependencies

Full training uses the existing Koa study installations, their frozen native
and portable data, saved epoch orders, source receipts, environment metadata,
and study configuration. The cluster entry point is `hpc/stage.sh` in each
source package; it is called with explicit study paths by the corresponding
Slurm submitter. These scripts are preserved for review, not presented as a
generic one-command installation on another machine.

CPU report entry points are `radius_source/code/report_radius.py` and
`operating_point_source/code/report_operating_points.py`. They read saved
predictions and verified attribution and do not train or run inference. They
still require the original frozen input artifacts and completed fit directories,
which are absent from this companion. Merely having these source files is not
sufficient to regenerate the full report.

Data arrays, model checkpoints, saved predictions, live job/run state, caches,
credentials, and generated test fixtures are not bundled. Immutable provenance
and reference metadata inside the original packages are retained because the
source packages depend on them. Historical absolute paths in provenance identify
the original runs and are not portable paths.

The dataset is not yet publicly released. The planned final release must provide
the data supporting reported experiments, fixed splits, labels, provenance,
preprocessing, and a documented tutorial/Colab. This companion is not a substitute
for that public-release requirement.

## Integrity check (standard-library Python only)

After extracting this ZIP, run from this directory:

```python
from pathlib import Path
import hashlib, json
root = Path('.')
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('All companion files match their recorded hashes.')
```

This check proves byte consistency with the packaged snapshot. It does not
rerun the science, establish detector realism, or replace the independent
calibration/evaluation still required after choosing the final procedure.
