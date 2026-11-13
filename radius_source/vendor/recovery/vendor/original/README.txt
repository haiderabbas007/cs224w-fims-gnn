FIMS PARALLEL RESEARCH V1
Retaining useful primary-track charge while identifying microcurler activity
Prepared 22 October 2026

WHAT THIS LAUNCH DOES

Three activities run independently where their inputs permit it:

1. Koa: report the completed small/wide CNN predictions across several operating
   points, immediately, without retraining. This is the first PARTIAL report.
2. Koa: train 12 geometric GNNs and four count/shape controls on the frozen
   development data, then produce a complete combined report.
3. WSL: recover continuous ideal-electron arrival times by replaying the exact
   existing readout on the original ROOT files. No Geant4 simulation is repeated.

Hit-level model training is the NEXT dependent stage, after its timestamp and
truth-alignment review passes. This package does not claim to train a hit model.
The site work does not wait for that export. There are no new ANN/SNN fits in
this launch; their existing results remain references for subsequent comparisons.

QUICK START

Save RUN_FIMS_Parallel_v1.sh directly in:
  C:\Users\Haider Abbas\Downloads\FIMS Research
Then in WSL:
  bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Parallel_v1.sh" start

The launcher installs its included, checksum-verified code, starts the local
readout replay, then transfers the small code package to Koa and submits jobs.
It reuses the capacity dataset already present on Koa. No 201 MB data upload
is needed. Complete password/Duo normally. One SSH authentication is reused
within each action. Keep the terminal open until cluster submission completes.

Check the cluster:
  bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Parallel_v1.sh" remote-status
Check the local arrival export:
  bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Parallel_v1.sh" hit-status
Download the latest cluster report (can be PARTIAL while fits run):
  bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Parallel_v1.sh" pull

Use start again after interruption. Verified finished fits/stages are reused;
active jobs are not duplicated. To start only one branch, use push (Koa) or
hit-export (WSL). The install action only installs/verifies code.

Once Slurm accepts the jobs, closing your laptop/SSH does not stop Koa work.
The WSL hit export needs WSL running; sleep/shutdown can pause or stop it. Its
verified completed shards are reused on restart. Graph training saves every
completed optimizer batch; an interrupted unfinished deterministic logistic
fit restarts, while its extracted features remain cached.

WHY THESE BRANCHES

A decision cutoff controls how readily a model flags observations. Lowering it
usually catches more microcurlers and also flags more primary activity. The
previous below-50% recall was at an unusually strict primary-retention target;
it was not the model's maximum recall. We will examine the tradeoff curve.

A graph model gives each site a local map and learns how neighboring positions
and times relate. Count controls tell us whether it learned more than how busy
that neighborhood is. Shape controls add compact descriptions of arrangement.
The existing CNN instead receives a raster patch. This compares implemented
pipelines, not architecture in isolation, because their encodings differ.

Continuous hit-level observations keep the simulated arrival time before it
is rounded into a 25 ns bin. Several electrons may occupy one site. A change
from sites to electrons also changes what one correct detection counts as;
therefore both native recall and common charge-weighted recall are required.

FIXED SITE EXPERIMENT

Data: existing 10,000-event common-source development production; 7,500 train,
2,500 validation; 5,044,099 occupied sites. All existing event partitions,
truth labels, ignored-site masks, empty events and microcurler-free events stay.
Previously inspected calibration/confirmation samples are NOT used as new tests.

Seeds: 20260924, 20260925, 20260926. Each neural model gets all three seeds.
Four GNN variants, 64 hidden channels:
  full2     2 message-passing layers, relative x/y/time attributes; 33,857 params
  full1     1 message-passing layer, same context; 12,929 params
  no_xy2    2 layers; zero x/y edge attributes, geometric connections retained
  current2  2 layers; current-bin context only

For the eight-bin variants, a query's graph contains the query and every site
on the same endcap within 2 mm, in bins t-7 through t. Each edge connects two
distinct selected sites within 2 mm; messages go from earlier/equal bins to
later/equal bins. Node inputs start at constant occupancy one. Relative x/y
and time offsets encode edge geometry. No absolute position, ancestry, source
identifier, target label or primary-charge attribution enters the encoder.

Every message layer stays within this query graph. Depth cannot silently expand
the observation window. Same-bin messages mean prediction happens after that
bin closes; this is not strict hit-by-hit inference. Sum aggregation retains
local occupancy. Geometric connectivity is retained in the no_xy2 ablation,
so that variant tests explicit spatial attributes, not all spatial information.

All models use the same frozen eligible-event orders, 10 epochs, 8-event
optimizer batches, class-balanced binary cross-entropy, Adam learning rate
0.001 and gradient clipping 1. Validation BCE selects the earliest minimum;
final-epoch results are secondary. GNN gradient microbatches do not alter that
optimizer-batch loss denominator. No threshold is used for checkpoint selection.

Four deterministic logistic controls are fitted once each, not repeated as
three artificial seeds: counts8, shape8, counts1, shape1. Counts use support
within 0.5/1/2 mm. Shape adds mean radius, radius spread and covariance
anisotropy. Every control uses the same causal context for its window, excludes
the query from neighborhood counts, includes ignored sites as context, scales
from supervised training rows only, and uses fixed L2=0.01.

The scientific comparison has no neighbor truncation. The implementation fails
rather than silently clipping a graph over its explicit memory guard. Report a
failed job if this happens so the implementation can be improved consistently.

RETENTION AND DETECTION REPORT

Reference retention targets: 99.95%, 99.9%, 99.5%, 99.0%. The stricter 99.95%
point is an extra reference; 99.5% and 99.0% are the relaxed candidates.
Also report primary-charge costs of reaching 80% and 90% microcurler-site recall.
Whole score ties are kept together; charge-loss budgets use integer counts.

Primary retention = 1 - primary charge at flagged sites / total primary charge.
Microcurler-site recall = supervised microcurler sites flagged / total such sites.
Microcurler-charge recall = attributed microcurler electrons flagged / total
attributed microcurler electrons, including charge at ignored mixed sites.
The report keeps these two recall denominators distinct.

Report precision, pooled primary loss, loss quantiles, the fraction of
primary-bearing events losing more than 5% charge, and per-event values.
Ignored sites receive predictions and their primary charge counts toward loss.
Pooled retention is not a guarantee that every event is protected. These are
observation/charge measures, not complete-track reconstruction efficiencies.

Thresholds in THIS report are fitted descriptively on development validation.
They are not a fresh confirmation and cannot establish deployment performance.
A final candidate and operating-point rule must be frozen before a new,
previously uninspected calibration/evaluation sample is used. A calibration
retention target is not a guarantee of identical evaluation retention.

CONTINUOUS-ARRIVAL EXPORT

Read the separate hit/README_HIT.txt for the replay contract. Existing saved
ideal-electron records contain bins, not precise timestamps. The export replays
the pinned original sampler, with unchanged RNG draws, and records the computed
float64 arrival times. Every old electron field, truth field, binary site and
mapping must match exactly before a shard is accepted. Bin jitter is never used.

Original ROOT source:
  ~/TPC-simulation/build/fims_expanded_transport_v2_bjaqa4o_/results
The launcher chooses the original g4tpc Python, requiring NumPy 2.5.0. It does
not activate Conda or modify that environment. An explicit FIMS_HIT_PYTHON or
FIMS_HIT_SOURCE override is supported, but identity checks still apply.

The arrival output contains ideal simulated electrons, not calibrated electronic
hits. Future hit models must use strictly earlier arrivals, with exact-time ties
sharing the same pre-update context. The original mixed-site supervision mask
is retained for the first paired comparison. A new labeling policy is a separate
experiment. An eight-bin site window is not exactly a rolling 200 ns hit window.
No electronics noise, deadtime or realistic background overlay is added here.

NEXT HIT-MODEL STUDY, AFTER EXPORT REVIEW

Keep the existing site branch and frozen split. First compare a small causal
hit baseline and a geometric hit model using unquantized relative arrival time,
then test longer temporal context as a declared ablation. Measure native hit
recall, common microcurler-charge recall, primary-charge loss and event damage
at the same set of retention targets. Choose hit-model size and batching from
the verified arrival-density audit, not from fabricated within-bin ordering.
The exporter provides the needed timestamps, truth sidecars and site mapping.

KOA EXECUTION

Existing capacity data:
  /mnt/lustre/koa/scratch/mhabbas/fims/cnn_capacity_v2/study_4cbf351c982f4fd6
New isolated studies:
  /mnt/lustre/koa/scratch/mhabbas/fims/parallel_v1/study_<package hash>

Koa's login-node home mount is noexec. Environment setup runs in a sandbox CPU
allocation, using the existing CUDA-built fm4npp Python. A private virtualenv
adds PyTorch Geometric 2.7.0; the existing environment remains unchanged.
CUDA availability and graph forward/backward tests run inside GPU allocations.

Default scheduling: 12 array tasks, up to three concurrent GPUs; one model/seed
per task. Each fit requests 1 GPU, 4 CPUs, 32 GB and an 8-hour limit. This is a
resource limit, not a runtime estimate. Queue delay and GNN throughput are not
measured yet. Resume requeues unfinished work and reuses verified checkpoints.
Count/shape controls and reports run on CPU. Report jobs depend on completed
fits. GPU availability does not remove graph-construction CPU work.

OUTPUTS

WSL automatically copies the continuous-arrival archive, small review ZIP and
summary to FIMS Research. Cluster reports need the pull action:
  fims_parallel_review_v1.zip
  fims_parallel_summary_v1.txt
  fims_parallel_operating_points_v1.png
  fims_parallel_retention_curves_v1.png
  plus CSV/JSON tables and full score-tie curves in the ZIP.
The exact report inventory is listed in its completion/status JSON. A PARTIAL
report never presents missing fits as completed or averages an incomplete seed
set as a three-seed result.

CS224W DELIVERABLES

New course work: graph construction, PyG MPNN, spatial/time/depth ablations,
controlled baseline comparison, error analysis, and a documented tutorial.
This package supplies executable research code. The final Colab/blog/public
release are later deliverables; they have not been published by this launch.

VALIDATION BEFORE DELIVERY

Synthetic checks: graph causality, permutation, exact windows, induced graph
boundaries, gradients, microbatch equivalence, optimizer/RNG recovery, masked
training-only scaling, controls, score ties, charge budgets, event damage,
partial reporting and scheduler duplicate/resume behavior.
Real data read-only checks: all 10,000 site geometries and frozen partitions;
all 20 preserved electron/site mappings; native validation charge attribution.
One training-only graph forward/backward was checked without fitting.
Actual ROOT replay and CUDA execution are launch-time gates. No new real-data
fit, cluster submission, timestamp replay or model-performance result is claimed
from the authoring environment.

Official implementation references:
  PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  Slurm arrays: https://slurm.schedmd.com/job_array.html
  Slurm submission: https://slurm.schedmd.com/sbatch.html
