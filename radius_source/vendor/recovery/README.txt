FIMS SITE-GNN MEMORY / WALL-TIME RECOVERY V1

WHAT THE STATUS LOG ESTABLISHES
The site-data preparation and all four count/shape controls completed.
Eight graph-network fits ran out of CUDA memory. Three reached their eight-hour
wall limit. One current-bin fit was still running in the supplied snapshot.
A fit's saved status.json can still say RUNNING after Slurm has ended the job.
The Slurm state is authoritative. The existing partial ZIP is not a final result.

THIS RECOVERY
Download REPAIR_FIMS_Parallel_GNN_v1.sh directly into:
C:\Users\Haider Abbas\Downloads\FIMS Research

In WSL:
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/REPAIR_FIMS_Parallel_GNN_v1.sh"

Check:
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/REPAIR_FIMS_Parallel_GNN_v1.sh" remote-status

Fetch the report (its summary explicitly says PARTIAL or COMPLETE):
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/REPAIR_FIMS_Parallel_GNN_v1.sh" pull

Use the same downloaded recovery launcher to resume after an interruption.
Password and Duo remain interactive; one authenticated SSH connection is reused
within each action. No training runs on the laptop or login node. No scientific
environment is installed or modified. Submitted jobs survive closing SSH/laptop.

WHAT CHANGES
1. Message MLP evaluation and destination summation are processed in chunks of
   16,384 edges. Nonreentrant activation checkpointing recomputes intermediate
   values during backward instead of retaining them for every edge.
2. A one-dimensional index_add replaces the expanded-index scatter path.
3. Jobs request a full NV-A30 GPU, with a 72-hour allocation limit. Up to 12 fits
   may run concurrently if Koa resources allow; queue time is not predictable.
   72 hours is an allowance, not a runtime estimate. If a job still reaches this
   limit, rerunning this launcher resumes saved work; continuation is not automatic.
4. Every allocated GPU must pass small-graph CUDA output/gradient parity and a
   synthetic 2,000,810-edge forward/backward memory check before real training.

WHAT IS PRESERVED
Original study and checkpoints remain in place. A separate versioned recovery
folder records the original source/protocol/checkpoint hashes. Unfinished fits
continue from captured model tensors, Adam moments, RNG state, epoch/batch
position and validation history. The only migrated checkpoint metadata field is
its protocol identity, which now records the execution change and its parent.
Completed original fits and four controls are verified and reused unchanged.
Any still-active original array is allowed to finish/timeout naturally; recovery
training waits for it and also checks original per-fit locks before snapshotting.

The fixed 7,500/2,500 event split, truth, graph neighborhoods, every node/edge,
64-channel parameters, initialization, 8-event optimizer batches, class-balanced
loss, fixed denominator, Adam settings, clipping and checkpoint selection remain
the same. No dense event or neighbor is removed. The original 2,000,000-edge
individual-graph fail-only ceiling remains; a larger real graph stops the fit
rather than being silently truncated.

Numerical summation grouping and GPU hardware can change floating-point roundoff.
The new execution is mathematically equivalent, not claimed bitwise identical
to the old CUDA implementation. Parent provenance and each runtime session are
included in the recovery lineage. Same-library/build checks reject silent
environment changes. The original code is bundled unmodified under vendor/.

SCIENTIFIC SCOPE
Development comparison only: full2, full1, no_xy2, current2; three fixed seeds.
All retention targets and full curves remain part of the report. No independent
confirmation data, new simulation, threshold retuning on a test set, or hit-model
training is introduced by this repair. Continuous-hit work remains a separate
preserved branch. Do not scale this GNN run to a larger sample before checking
the repaired pilot's memory, runtime and results.

VALIDATION PERFORMED LOCALLY
PyTorch 2.7.0 CPU / NumPy 1.26.4 / SciPy 1.15.2 / PyG 2.7.0.
Six memory-model tests: all four variants, float32/float64 logits and gradients,
initialization/state keys, no-edge behavior, constant inputs, and activation
storage. Invented 9,900-edge example: saved intermediate storage 13,129,056 bytes
original vs 179,856 bytes recovery; this is not measured total GPU memory.
Trainer tests: all four models' fixed-D 8-event Adam updates, exact continuation
within the repaired runtime, source-state preservation, idempotent migration,
completed-fit reuse, active-source refusal and corrupted-state rejection.
Eleven scheduler tests: queue failures, duplicate prevention, afterany/afterok
dependencies, resource arguments, exact report cancellation and absolute paths.
Control snapshot integrity, package roundtrip, shell syntax and imports checked.
Actual Koa CUDA and real-data execution have NOT been performed here; they are
gated on the cluster. No completion time or successful full run is guaranteed.

DOCUMENTATION CHECKED
Koa GPU request syntax (full A30):
https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407800/How+to+Request+GPUs+on+Koa
Koa GPU partition maximum wall time (three days):
https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa
PyTorch 2.7 activation checkpointing:
https://docs.pytorch.org/docs/2.7/checkpoint.html
Slurm dependencies and pre-timeout signals:
https://slurm.schedmd.com/sbatch.html
