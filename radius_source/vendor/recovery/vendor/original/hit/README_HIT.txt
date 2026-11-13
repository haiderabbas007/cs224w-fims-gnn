FIMS continuous ideal-electron arrival export v1

Why this stage exists
The preserved ideal-electron files contain 25 ns bins, not unquantized arrival
times. The pinned readout computes a floating-point arrival time and then bins
it. This exporter replays that same deterministic readout on the same ROOT
events and retains that computed value. It does not jitter a bin, invent an
arrival sequence, run Geant4 again, or change the original site experiment.

Inputs
The frozen 10,000-event development production, normally:
  ~/TPC-simulation/build/fims_expanded_transport_v2_bjaqa4o_/results
The source manifest, fixed 7,500/2,500 split, ROOT files, original readout
archives, per-deposition truth, RNG seeds, and producer digest are verified.
The protocol and split-lock hashes must match the pinned production manifest;
their freeze and split links are checked before replay and again at publication.
Use the original g4tpc interpreter: the captured NumPy version is 2.5.0.
Python, NumPy and uproot versions are recorded. NumPy mismatch stops replay.

Implementation
readout_v2_reference.py is the exact captured readout_adapter_v2.py, SHA256:
c730d745ebe8870661693d0c74e99dc1ac8c7860b02517f9033d1ab8d621bce3
Only an explicit set of pure definitions is compiled. Two AST additions retain
arrival[accepted][order] in the per-event token and float64 output column.
Random draws, random seed keys, arithmetic, acceptance, order, truth rules,
and original aggregation are unchanged. The reference file is never patched.

Before a shard is published
1. Recompute source truth and compare it to the native truth.
2. Replay every original electron observable and truth field exactly.
3. Reconstruct every binary-site observable, label and electron-to-site map.
4. Require floor(arrival_time_ns / 25) == original absolute bin for every row.
5. Recheck input hashes and write atomic outputs followed by complete.json.
All empty and microcurler-free events and original masks are retained.

Outputs and timing policy
continuous_inputs_v1.npz: physical pixel/endcap fields, event pointers and
float64 arrival_time_ns. It intentionally contains no ancestry or truth.
continuous_targets_v1.npz: original labels/masks, electron-to-site mapping,
primary charge weights and an observation-derived arrival-order permutation.
site_inputs_v1.npz / site_targets_v1.npz: byte-identical original site files.
Rows stay aligned with the native electron archives. Storage order is NOT
chronology. Sort by arrival_time_ns; future hit models should use strictly
earlier arrivals and process exact-time ties from the same pre-update context.
The original mixed-25-ns-site supervision mask remains in use initially. A
future revised collision-label policy must be its own declared experiment.

This is still an ideal sampled-electron model, not calibrated hardware hit
timing. Each step uses post-step Gtime, event boundaries are known, and no
electronics noise/deadtime has been introduced. Continuous within-event time
does not establish a fully overlaid continuous detector stream.

Run / resume (the outer downloaded launcher supplies these paths)
  python export_continuous_hits_v1.py --self-test
  python export_continuous_hits_v1.py --source-dir SOURCE --output-dir NEW_OUTPUT
The Python worker owns NEW_OUTPUT/worker.lock. Resume skips only completed
shards whose source, code, environment and output hashes still agree.

Completed portable archive, review ZIP and summary are copied automatically to:
  C:\Users\Haider Abbas\Downloads\FIMS Research
On failure, a small diagnostic ZIP is copied there. No original files change.

Validation performed during authoring
- Synthetic replay and original-array parity, signed ignored labels, collisions,
  empty events, both endcaps, real sub-bin values, exact ties, truth independence,
  chronological permutation and atomic NPZ roundtrip: PASS.
- Preserved real input schema and electron/site mapping across all 20 shards,
  10,000 development events and 5,150,745 accepted ideal electrons: PASS.
- Frozen protocol/split-lock hashes and lineage links, with deliberate metadata
  tampering rejected before replay: PASS.
- Actual continuous timestamps cannot be checked here without the source ROOT
  files. The full replay gate runs on WSL; no hit-model training result is claimed.
