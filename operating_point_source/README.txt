FIMS OPERATING-POINT REVIEW V1
Prepared 12 November 2026

Purpose
Use the 22 completed selected fits from the previous parallel study to compare
99%, 95% and 90% pooled primary-charge retention. Keep 99.9% and 99.5% as
historical references. Show classifier-score overlap and the distribution of
charge loss within individual events. This is reporting from saved predictions:
no model training, model inference, simulation, or radius-job changes.

What is included
Six CNN fits (small/wide, three seeds each), twelve GNN fits (two layers, one
layer, no explicit spatial offsets, current-bin only; three seeds each), and
four deterministic count/shape controls. Older ANN/SNN and event-balanced fits
are not part of this particular 22-fit input archive.

All fits use the same 2,500 development-validation events. Each neural fit keeps
its previously selected checkpoint; no new checkpoint selection occurs here.
Thresholds are chosen on these same validation events. These results are
development comparisons, not fresh confirmation or deployment guarantees.

Download and run
Save RUN_FIMS_Operating_Point_Review_v1.sh directly in:
C:\Users\Haider Abbas\Downloads\FIMS Research

From WSL:
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Operating_Point_Review_v1.sh"

Complete Koa password/Duo normally. One connection is reused during an action.
The launcher transfers the small code package and submits a shared-CPU report
job. It does not wait for the radius experiment or request a GPU. Once submitted,
the batch job continues after the laptop or SSH connection closes.

Check:
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Operating_Point_Review_v1.sh" status

Download when COMPLETE:
bash "/mnt/c/Users/Haider Abbas/Downloads/FIMS Research/RUN_FIMS_Operating_Point_Review_v1.sh" pull

The verified review ZIP, summary and report files go to the same FIMS Research
folder. Rerunning the launcher retains an existing live report job or verifies
an already completed report; it does not submit duplicate training.

Interpretation
Positive means a supervised microcurler site, not an entire simulated event.
Site recall counts sites. Microcurler-charge recall counts accepted attributed
electrons, including microcurler charge on mixed/ignored sites. These are distinct.
Primary-charge loss includes primary attribution on ALL scored sites, even sites
excluded from supervised loss. Whole groups of identical scores are kept together;
cutoffs are selected using integer primary-electron budgets, without interpolation.
Actual achieved retention can exceed the target because charge is discrete and
score ties are not split.

Equal-event mean microcurler recall averages only events with supervised positive
sites; empty and microcurler-free events remain in the population and in applicable
primary-protection metrics. Event damage is measured separately: an event's lost
primary charge divided by its own primary charge. The report includes the fraction
of primary-bearing events losing more than 5%, other tails and loss quantiles.
The 5% line is a diagnostic, not an approved physics acceptance criterion.

Classifier scores are not claimed to be calibrated probabilities. Histograms
include supervised microcurler sites, supervised other sites, and a separate
primary-charge-weighted distribution including ignored sites. Each seed retains
its own cutoff; no artificial shared cutoff is constructed across models/seeds.
The source paper/code reference for probability calibration is:
https://scikit-learn.org/stable/modules/calibration.html

Seed ranges describe variation between fits on the same data; they are not
independent-data confidence intervals. GNN/CNN encoding and runtime differences
from the original comparison remain documented. Geometry, labels and original
saved predictions are not altered to obtain a better result.

Before use, source data, selected predictions, protocols and completion records
are checked against frozen identities. Changed or missing inputs cause an error.
Results are written to a new study directory. Report-only interruption can be
rerun; training checkpoints and completed radius controls remain untouched.

Next simulation work
reference/NEXT_CAMPAIGN_SETTINGS.json and NEXT_CAMPAIGN_NOTES.txt record Peter's
approved proof-of-principle transport assumptions and the proposed next pilot.
They are a design record, not a claim that new simulation or continuous-hit
training has run. Current results retain their original detector settings.

Runtime
This is CPU postprocessing of saved outputs. The job has a four-hour ceiling,
not a predicted runtime. Queue delay and shared-filesystem verification time are
unknown; status/log progress is the evidence for an estimate after it starts.
