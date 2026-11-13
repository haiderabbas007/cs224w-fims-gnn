FIMS GNN RADIUS SENSITIVITY V1
Prepared 12 November 2026. Development experiment; not submitted by this assistant.

WHAT THIS ANSWERS
Does the GNN identify microcurlers better when it sees activity within 1, 2 or
4 mm of the observation being classified? We compare detection against both
overall primary-charge loss and charge loss concentrated in individual events.
This directly addresses the CS224W proposal feedback about neighborhood radius.

WHAT WILL RUN
Nine fresh one-layer GNN fits: three radii x three seeds (20260924/25/26).
Each has 64 hidden channels and 12,929 trainable parameters. Six deterministic
controls run alongside them: counts-only and counts-plus-shape at each radius.
The controls use the same observations as the corresponding GNN.

The existing 10,000 development events are reused: 7,500 training and 2,500
validation. Labels, frozen epoch orders, optimizer, loss, 10-epoch budget and
eight-bin history are unchanged. The time bins remain 25 ns. Same-bin context
requires waiting for that bin to close. Event boundaries remain known in this
development branch. The model never receives truth ancestry or primary charge.

The query observation radius and scientific edge radius change together.
Relative coordinates still use the original fixed 2 mm normalization, so the
experiment does not also change coordinate scale. The control count radii are
R/4, R/2 and R; their shape normalization also stays fixed at 2 mm.

The one-layer GNN was chosen from the earlier development comparison. Historical
2 mm fits changed numerical execution during memory recovery, so all nine fits
start fresh with one matched implementation. Historical models are not retrained,
overwritten or silently mixed into the new averages.

For one layer, only messages ending at the query can affect its prediction.
The runner avoids computing messages to other destinations, retains every window
node and every relevant incoming edge, and tests prediction, gradient and Adam
update equivalence against the full graph. No neighbor sampling or truncation
is used. This shortcut must not be extended to a two-layer GNN without a new
derivation. This study measures neighborhood-context radius; it does not
independently measure neighbor-to-neighbor connectivity.

INSTALL AND SUBMIT FROM WSL
Download RUN_FIMS_GNN_Radius_Sensitivity_v1.sh directly into:
C:\Users\Haider Abbas\Downloads\FIMS Research

bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_GNN_Radius_Sensitivity_v1.sh"

Complete the usual Koa password and Duo prompts. One authenticated SSH connection
is reused for each action. The launcher transfers only the small code package;
the frozen event data already on Koa are read in place. It uses the existing
scientific Python environment, without installing or changing that environment.
Python runs in a compute allocation to avoid the login node's noexec home mount.

Wait for the "Submitted" job IDs and final submission message before closing
the terminal. After Slurm accepts the jobs, the laptop and SSH connection are
not needed. If the initial sandbox allocation is still queued, submission has
not finished yet.

The default is up to nine independent GPU fits, each requesting one NV-A30,
4 CPUs and 32 GB RAM. Three CPU tasks run the six controls. The report waits
for these jobs. Each GPU allocation has a 72-hour limit; that is not a runtime
estimate. Actual runtime and queue delay are unknown until Koa starts the jobs.

CHECK PROGRESS
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_GNN_Radius_Sensitivity_v1.sh" status

RESUME AFTER A JOB INTERRUPTION
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_GNN_Radius_Sensitivity_v1.sh" resume

The submitter retains active jobs, skips verified complete fits, and submits
missing work. Model, optimizer and random-number state are saved after every
training batch. An interrupted deterministic control reuses its verified feature
cache and completed fit; its unfinished optimizer fit may need to restart.
Changing the package, source data or saved fit identity causes a clear failure
instead of silently mixing results. No completed prior experiment is altered.

DOWNLOAD THE REPORT
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_GNN_Radius_Sensitivity_v1.sh" pull

The review ZIP, summary, tables and plots go directly into FIMS Research.
Upload fims_radius_sensitivity_review_v1.zip for interpretation. A report after
a failed task is labelled PARTIAL and lists missing seeds; it is not a finished
comparison. Resume missing work, then pull again after the final report completes.

REPORT CONTENTS AND LIMITS
Selected checkpoints use the earliest minimum validation BCE. Final-epoch
results are reported separately. The targets are 99.95%, 99.9%, 99.5%, 99.0%,
95% and 90% pooled primary-charge retention. Whole equal-score groups stay
together when setting each cutoff. Actual achieved retention is reported.

Primary endpoints:
  - Pooled supervised microcurler-site recall.
  - Equal-event mean recall over events containing supervised microcurler sites.
  - Pooled primary-charge retention, including charge at ignored-label sites.
  - Fraction of primary-bearing events losing strictly more than 5% of their
    own primary charge.
Precision, microcurler-charge recall, event-loss tails, all three neural seeds,
matched-seed differences from 2 mm, and execution costs are also recorded.

Cutoffs are selected on the same development validation sample used for these
plots. These are developmental comparisons, not independent confirmation.
Seed ranges are not confidence intervals. The detector's acceptable loss is
still a physics question. Charge retention does not establish track-finding
efficiency or momentum/direction resolution.

Detector upgrades, continuous-hit training, larger simulation campaigns and
noise/background additions are on hold. This run creates no new physics events
and reads no independent confirmation sample.

VERIFICATION
The package includes synthetic CPU tests for graph causality, endcap and time
boundaries, permutation invariance, truth exclusion, full-graph forward/gradient/
optimizer equivalence, exact checkpoint resume, historical 2 mm control parity,
metric denominators and score ties, report completeness, scheduler behavior and
archive integrity. These tests do not establish real detector performance.
Actual CUDA parity and frozen-data checks run on Koa before each real fit.
No real Koa job or CUDA run was performed while preparing this download.

PROVENANCE
The package preserves the original and memory-recovery source packages byte for
byte under vendor/recovery. Their manifests, the portable-data lineage and the
reviewed historical full1 protocols/completion files are pinned. New outputs use
a separate versioned directory under:
/mnt/lustre/koa/scratch/mhabbas/fims/gnn_radius_sensitivity_v1
EXPERIMENT_PLAN.json and PACKAGE_MANIFEST.json are the machine-readable records.
