#!/usr/bin/env python3
"""V2 transport adapter; truth, sampling, aggregation and encoding remain v1.

Only the production validator and provenance annotations are adapted. The exact
original is shipped as frozen_readout_v1.py; sensitive function ASTs are compared
in the adapter tests. A new run UID intentionally gives a new readout RNG stream.

Whole simulation productions have their roles assigned before generation. No model
is fitted or scored. Input NPZs contain physical addresses/times, event boundaries
and readout metadata only; truth and run provenance remain in separate sidecars.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import numpy as np
import awkward as ak
import uproot
import production_adapter_v2 as production
import zipfile

PREFIX = "fims_common_source"
PARTITION_SPECS = {'block00': {'events': 500, 'seeds': [2026100601, 2026100602], 'readout_seed': 2026100701}, 'block01': {'events': 500, 'seeds': [2026100603, 2026100604], 'readout_seed': 2026100702}, 'block02': {'events': 500, 'seeds': [2026100605, 2026100606], 'readout_seed': 2026100703}, 'block03': {'events': 500, 'seeds': [2026100607, 2026100608], 'readout_seed': 2026100704}, 'block04': {'events': 500, 'seeds': [2026100609, 2026100610], 'readout_seed': 2026100705}, 'block05': {'events': 500, 'seeds': [2026100611, 2026100612], 'readout_seed': 2026100706}, 'block06': {'events': 500, 'seeds': [2026100613, 2026100614], 'readout_seed': 2026100707}, 'block07': {'events': 500, 'seeds': [2026100615, 2026100616], 'readout_seed': 2026100708}, 'block08': {'events': 500, 'seeds': [2026100617, 2026100618], 'readout_seed': 2026100709}, 'block09': {'events': 500, 'seeds': [2026100619, 2026100620], 'readout_seed': 2026100710}, 'block10': {'events': 500, 'seeds': [2026100621, 2026100622], 'readout_seed': 2026100711}, 'block11': {'events': 500, 'seeds': [2026100623, 2026100624], 'readout_seed': 2026100712}, 'block12': {'events': 500, 'seeds': [2026100625, 2026100626], 'readout_seed': 2026100713}, 'block13': {'events': 500, 'seeds': [2026100627, 2026100628], 'readout_seed': 2026100714}, 'block14': {'events': 500, 'seeds': [2026100629, 2026100630], 'readout_seed': 2026100715}, 'block15': {'events': 500, 'seeds': [2026100631, 2026100632], 'readout_seed': 2026100716}, 'block16': {'events': 500, 'seeds': [2026100633, 2026100634], 'readout_seed': 2026100717}, 'block17': {'events': 500, 'seeds': [2026100635, 2026100636], 'readout_seed': 2026100718}, 'block18': {'events': 500, 'seeds': [2026100637, 2026100638], 'readout_seed': 2026100719}, 'block19': {'events': 500, 'seeds': [2026100639, 2026100640], 'readout_seed': 2026100720}}
PRODUCTION_SCHEMA = production.SCHEMA
ALL_BRANCHES = "charge driftTime_notSmeared StepLength MomIn MassIn Gtime HitPathLength_fromTrack prePos_x prePos_y prePos_z postPos_x postPos_y postPos_z processName EventID Pt Pz prePos_x_smeared prePos_y_smeared prePos_z_smeared driftTime_smeared sigmaT_mm sigmaL_mm TrackID ParentID PDG ParticleName CreatorProcess".split()
STRING_BRANCHES = {"processName", "ParticleName", "CreatorProcess"}
INT_BRANCHES = {"EventID", "TrackID", "ParentID", "PDG"}
TRUTH_RULE = "PDG=11, ParentID!=0, 0.100<=first recorded MomIn<=0.900 MeV/c; first=min recorded path with stable ROOT-row tie; no creator restriction; independently qualifying core overrides descendant; all zero-charge rows retained for ancestry"
OUTPUTS = [PREFIX+"_"+n for n in ("hit_truth_labels_v1.npz", "ideal_electron_inputs_v1.npz", "ideal_electron_targets_v1.npz", "binary_spike_inputs_v1.npz", "binary_spike_targets_v1.npz", "event_audit_v1.csv", "track_audit_v1.csv", "readout_audit_v1.json", "readout_audit_v1.txt")]


def validate_production_manifest(manifest, freeze):
    """Require the explicitly versioned transport identity and original shard schedule."""
    require(manifest.get("schema") == PRODUCTION_SCHEMA and manifest.get("status") == "PASS", "Need successful v2 production")
    require(freeze.get("schema") == "fims_expanded_transport_protocol_v2", "Wrong v2 frozen plan")
    require(freeze.get("source_sha256") == production.TRIAL_SOURCE_HASHES, "Frozen trial source changed")
    require(freeze.get("transport") == production.TRANSPORT_IDENTITY, "Frozen transport identity changed")
    part = manifest.get("partition")
    require(part in PARTITION_SPECS, "Unexpected shard")
    expected = PARTITION_SPECS[part]
    planned = freeze["partitions"][part]
    require(planned["events"] == expected["events"] and planned["production_seeds"] == expected["seeds"]
            and planned["readout_seed"] == expected["readout_seed"], "Frozen production/readout schedule changed")
    spec = production.partition_spec(part, manifest["freeze_sha256"], production.TRIAL_SOURCE_HASHES)
    require(all(manifest.get(k) == v for k,v in spec.items()) and manifest.get("production_spec") == spec, "Production spec changed")
    uid = production.run_uid(spec)
    require(manifest.get("run_uid") == uid and manifest.get("root_name") == production.root_name(part), "Wrong versioned identity")
    require(manifest.get("copied_binary_sha256") == production.TRANSPORT_IDENTITY["binary_sha256"], "Wrong transport binary")
    require(manifest.get("transport_environment") == production.PRODUCTION_ENV, "Wrong transport runtime environment")
    require(manifest.get("source_edits") is True and manifest.get("actual_source_edits") is True
            and manifest.get("original_repository_edits") is False and manifest.get("build_performed") is False,
            "Incorrect transport/source edit provenance")
    return uid, expected["events"]


def verify_transport_artifacts(folder, manifest):
    """Verify actual copied binary, source snapshot, macro and logged runtime settings."""
    require(production.sha(production.HERE/"frozen_source_validator_v1.py") == production.REFERENCE["original_validator_sha256"], "Frozen source validator changed")
    require(production.sha(production.HERE/"frozen_readout_v1.py") == production.REFERENCE["original_readout_sha256"], "Frozen sampler reference changed")
    require(sha256(folder/"TPC") == manifest["copied_binary_sha256"], "Copied executable changed")
    snapshot = folder/"source_snapshot.zip"
    require(sha256(snapshot) == manifest["source_snapshot_sha256"], "Source snapshot changed")
    with zipfile.ZipFile(snapshot) as archive:
        for name,digest in production.TRIAL_SOURCE_HASHES.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Snapshot source changed: "+name)
    macro = folder/f"fresh_{manifest['partition']}_500.mac"
    require(sha256(macro) == manifest["macro_sha256"] and macro.read_text() == production.macro_text(manifest["partition"]), "Production macro changed")
    actual = production.validate_log((folder/"simulator.log").read_text(errors="replace"), manifest["partition"])
    require(actual == manifest["actual_transport_log"], "Logged actual transport differs from manifest")
    return {"transport": production.TRANSPORT_IDENTITY, "actual_source_edits": True,
            "original_repository_edits": False, "readout_adapter_sha256": sha256(__file__),
            "frozen_readout_reference_sha256": production.REFERENCE["original_readout_sha256"],
            "run_key_policy": "New v2 UID changes per-event sampler RNG key; sampling algorithm and configured seeds are unchanged"}


def load_production(folder, freeze_file):
    manifest_path = folder/"production_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(Path(freeze_file).read_text())
    require(sha256(freeze_file) == manifest["freeze_sha256"], "Frozen plan hash differs from production manifest")
    uid, event_count = validate_production_manifest(manifest, freeze)
    transport_provenance = verify_transport_artifacts(folder, manifest)
    root_path = folder/manifest["root_name"]
    require(sha256(root_path) == manifest["root_sha256"] and root_path.stat().st_size == manifest["root_bytes"],
            "ROOT hash/size differs from manifest")
    with uproot.open(root_path) as f:
        require("HitNtuple" in f and "EventNtuple" in f, "Required ROOT trees missing")
        ht, et = f["HitNtuple"], f["EventNtuple"]
        require(et.num_entries == event_count and set(ALL_BRANCHES).issubset(ht.keys()), "Event count or hit branches differ")
        root = {}
        for name in ALL_BRANCHES:
            a = ht[name].array(library="ak")
            if name in STRING_BRANCHES:
                values = ak.to_list(a)
                require(len(values) == ht.num_entries and all(isinstance(v, str) for v in values), "Invalid scalar string branch: "+name)
                root[name] = np.asarray(values, dtype=str)
            else:
                a = ak.to_numpy(a)
                require(a.ndim == 1 and len(a) == ht.num_entries and a.dtype.kind in "iuf" and np.isfinite(a).all(), "Invalid scalar numeric branch: "+name)
                root[name] = integer_array(a, name) if name in INT_BRANCHES else a.astype(np.float64)
        require(np.all((root["EventID"] >= 0) & (root["EventID"] < event_count)), "Hit event outside production run")
        counts = np.bincount(root["EventID"], minlength=event_count)
        event_hits = ak.to_numpy(et["NumberOfHits"].array(library="ak"))
        # Audited single-worker generation writes EventNtuple in EventID order.
        require(np.array_equal(event_hits, counts), "Ordered event/hit ntuple counts disagree")
    checked, residuals = validate_root_arrays(root, uid)
    root.update(checked)
    require(np.all(root["TrackID"] > 0) and np.all(root["ParentID"] >= 0), "Invalid track identity")
    for name in ("MomIn", "HitPathLength_fromTrack", "StepLength", "MassIn", "driftTime_smeared"):
        require(np.all(root[name] >= 0), "Negative value in "+name)
    require(len(root["EventID"]) == manifest["results"]["hit_rows"], "Manifest hit count differs")
    return root, manifest, residuals, {"manifest_sha256": sha256(manifest_path), "root_sha256": sha256(root_path),
                                       "freeze_sha256": sha256(freeze_file), **transport_provenance}


def label_tracks(root, event_count, uid):
    n = len(root["EventID"])
    role = np.zeros(n, dtype=np.uint8)
    tracks = []
    for event in range(event_count):
        event_rows = np.flatnonzero(root["EventID"] == event)
        info, parents = {}, {}
        for tid in np.unique(root["TrackID"][event_rows]):
            rows = event_rows[root["TrackID"][event_rows] == tid]
            for name in ("ParentID", "PDG", "ParticleName", "CreatorProcess"):
                require(np.all(root[name][rows] == root[name][rows[0]]), f"E{event} T{tid}: inconsistent {name}")
            first = int(rows[np.argmin(root["HitPathLength_fromTrack"][rows])])
            parent, pdg = int(root["ParentID"][first]), int(root["PDG"][first])
            require(parent != tid, f"E{event} T{tid}: self-parent")
            parents[int(tid)] = parent
            info[int(tid)] = {"rows": rows, "first": first, "parent": parent, "pdg": pdg,
                              "p": float(root["MomIn"][first])}
        ancestry = {}
        for tid in info:
            chain, current = [], parents[tid]
            while current != 0:
                require(current in info, f"E{event} T{tid}: missing recorded ancestor {current}; no silent ancestry guess")
                require(current != tid and current not in chain, f"E{event} T{tid}: ancestry cycle")
                chain.append(current); current = parents[current]
            ancestry[tid] = chain
        core = {tid for tid, r in info.items() if r["pdg"] == 11 and r["parent"] != 0 and .100 <= r["p"] <= .900}
        for tid, r in info.items():
            ancestor_core = any(a in core for a in ancestry[tid])
            code = 1 if tid in core else 2 if ancestor_core else 0
            role[r["rows"]] = code
            tracks.append({"run_uid": uid, "EventID": event, "TrackID": tid, "ParentID": r["parent"],
                           "PDG": r["pdg"], "ParticleName": str(root["ParticleName"][r["first"]]),
                           "CreatorProcess": str(root["CreatorProcess"][r["first"]]),
                           "first_recorded_momentum_MeV_c": r["p"], "first_recorded_hit_row": r["first"],
                           "hit_rows": len(r["rows"]), "positive_charge_rows": int(np.sum(root["charge"][r["rows"]] > 0)),
                           "truth_class": code, "has_core_ancestor": ancestor_core,
                           "core_overrides_descendant": tid in core and ancestor_core})
    return role, tracks


def verify_candidates(folder, manifest, tracks):
    cores = {(r["EventID"], r["TrackID"]): r for r in tracks if r["truth_class"] == 1}
    require(len(cores) == manifest["results"]["provisional_secondary_core_tracks"], "Core-track count differs from production audit")
    require(len({k[0] for k in cores}) == manifest["results"]["provisional_secondary_core_events"], "Core-event count differs from production audit")
    path = folder/"candidate_track_audit.csv"
    if not path.exists():
        return {"candidate_csv_present": False}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [(int(r["EventID"]), int(r["TrackID"])) for r in rows]
    require(len(set(keys)) == len(keys) and set(keys) == set(cores), "Candidate CSV track identities differ")
    for key, old in zip(keys, rows):
        new = cores[key]
        require(old["run_uid"] == manifest["run_uid"], "Candidate CSV belongs to another production run")
        for col in ("ParentID", "hit_rows", "positive_charge_rows"):
            require(int(old[col]) == new[col], "Candidate CSV differs in "+col)
        require(np.isclose(float(old["first_recorded_momentum_MeV_c"]), new["first_recorded_momentum_MeV_c"], rtol=1e-12, atol=1e-14), "Candidate momentum differs")
    return {"candidate_csv_present": True, "candidate_csv_sha256": sha256(path), "candidate_csv_validation": "PASS"}


def export_electrons(root, role, event_count, uid, seed, budget, partition):
    require(partition in PARTITION_SPECS, "Invalid preassigned production partition")
    ikeys = {"global_pixel_x": np.int16, "global_pixel_y": np.int16, "readout_side": np.uint8, "time_bin_abs": np.int64, "time_bin_rel": np.int64}
    tkeys = {"source_hit_row": np.int64, "truth_class": np.uint8, "target_core": np.int8, "supervision_mask": bool,
             "ambiguous_site_mask": bool, "hit_event_index": np.int32, "track_id": np.int64, "parent_id": np.int64, "pdg": np.int64}
    iparts, tparts = {k: [] for k in ikeys}, {k: [] for k in tkeys}
    ptr, audits, generated = [0], [], 0
    for event in range(event_count):
        rows = np.flatnonzero(root["EventID"] == event)
        token, audit = sample_event(root, role, rows, seed, uid, event, budget-generated)
        audit["run_uid"] = audit.pop("sample")
        generated += audit["generated_electrons"]
        target, mask, ambiguous, collisions = make_targets_and_collision_mask(token)
        audit.update(collisions)
        audit["raw_roles_other_core_descendant"] = np.bincount(role[rows], minlength=3).tolist()
        audit["electron_supervised_other_core"] = [int(np.sum(mask & (target == c))) for c in (0, 1)]
        n = len(target); ptr.append(ptr[-1]+n)
        values = {**token, "target_core": target, "supervision_mask": mask, "ambiguous_site_mask": ambiguous,
                  "hit_event_index": np.full(n, event, np.int32), "track_id": root["TrackID"][token["source_hit_row"]],
                  "parent_id": root["ParentID"][token["source_hit_row"]], "pdg": root["PDG"][token["source_hit_row"]]}
        for k in ikeys: iparts[k].append(token[k])
        for k in tkeys: tparts[k].append(values[k])
        audits.append(audit)
    ptr = np.asarray(ptr, np.int64)
    origins = np.asarray([r["event_time_origin_bin_abs"] for r in audits], np.int64)
    model = {"schema_version": np.asarray(PREFIX+"_ideal_electron_inputs_v1"), "event_ptr": ptr,
             "event_time_origin_bin_abs": origins, "pixel_pitch_mm": np.asarray(PIXEL_MM), "clock_ns": np.asarray(CLOCK_NS),
             "grid_pixels_per_side": np.asarray(GLOBAL_PIXELS), "readout_side_names": np.asarray(["negative_z", "positive_z"]),
             "model_assumptions_json": np.asarray(json.dumps(ELECTRON_ASSUMPTIONS)),
             **{k: concatenate(iparts[k], dt) for k, dt in ikeys.items()}}
    truth = {"schema_version": np.asarray(PREFIX+"_ideal_electron_targets_v1"), "event_ptr": ptr.copy(),
             "event_time_origin_bin_abs": origins.copy(), "run_uid": np.asarray(uid), "event_run_uid": np.full(event_count, uid),
             "event_id_raw": np.arange(event_count, dtype=np.int64), "partition_status": np.asarray(partition),
             **{k: concatenate(tparts[k], dt) for k, dt in tkeys.items()}}
    validate_electrons(model, truth)
    return model, truth, audits


def validate_electrons(x, y):
    require(str(x["schema_version"]) == PREFIX+"_ideal_electron_inputs_v1", "Wrong common-source electron schema")
    ptr = x["event_ptr"]; n = int(ptr[-1]); ne = len(ptr)-1
    require(ptr[0] == 0 and np.all(np.diff(ptr) >= 0) and np.array_equal(ptr, y["event_ptr"]), "Electron pointer mismatch")
    require(str(y["partition_status"]) in PARTITION_SPECS and "event_split" not in y and "split_names" not in y, "Unexpected partition assignment")
    require(y["event_id_raw"].tolist() == list(range(ne)) and len(y["event_run_uid"]) == ne
            and np.all(y["event_run_uid"] == str(y["run_uid"])), "Event identity mismatch")
    require(not any(k in x for k in ("truth_class", "target_core", "source_hit_row", "run_uid", "event_split", "partition_status")), "Truth/provenance in model inputs")
    for k in ("global_pixel_x", "global_pixel_y", "readout_side", "time_bin_abs", "time_bin_rel"):
        require(x[k].shape == (n,), "Wrong electron input shape: "+k)
    for k in ("truth_class", "target_core", "supervision_mask", "ambiguous_site_mask", "parent_id", "track_id", "pdg", "source_hit_row", "hit_event_index"):
        require(y[k].shape == (n,), "Wrong electron truth shape: "+k)
    role = y["truth_class"]
    require(np.isin(role, [0, 1, 2]).all(), "Invalid electron role")
    target = role.astype(np.int8, copy=True); target[role == 2] = -1
    require(np.array_equal(y["target_core"], target) and np.array_equal(y["supervision_mask"], (role != 2) & ~y["ambiguous_site_mask"]), "Electron role/mask mismatch")
    for k in ("global_pixel_x", "global_pixel_y"):
        require(np.all((x[k] >= 0) & (x[k] < GLOBAL_PIXELS)), "Invalid pixel")
    require(np.isin(x["readout_side"], [0, 1]).all(), "Invalid endcap")
    for e, (a, b) in enumerate(zip(ptr[:-1], ptr[1:])):
        t = x["time_bin_abs"][a:b]; origin = x["event_time_origin_bin_abs"][e]
        require(np.all(np.diff(t) >= 0) and np.all(t >= 0) and np.all(y["hit_event_index"][a:b] == e), "Time/event ordering mismatch")
        require(origin == (int(t[0]) if len(t) else -1) and np.array_equal(x["time_bin_rel"][a:b], t-origin), "Empty event or time-origin mismatch")


def save_npz(path, arrays):
    # Publish only a fully closed, verified archive; never expose a partial ZIP.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".npz_stage_", suffix=".npz", delete=False) as stream:
        stage = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stage.unlink(missing_ok=True)
            raise
    try:
        with np.load(stage, allow_pickle=False) as z:
            require(set(z.files) == set(arrays) and all(np.array_equal(z[k], arrays[k]) for k in arrays), "NPZ serialized roundtrip mismatch")
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def write_csv(path, rows):
    require(bool(rows), "No audit rows to write")
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main_run(args):
    require(args.seed >= 0 and args.max_electrons > 0, "Invalid seed or electron budget")
    folder, out = args.production_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS: require(not (out/name).exists(), "Refusing to overwrite "+str(out/name))
    root, manifest, checks, provenance = load_production(folder, args.freeze_file)
    partition = manifest["partition"]
    require(args.seed == PARTITION_SPECS[partition]["readout_seed"], "Sampler seed differs from frozen partition seed")
    uid, ne = manifest["run_uid"], int(manifest["events"])
    role, tracks = label_tracks(root, ne, uid)
    provenance.update(verify_candidates(folder, manifest, tracks))
    provenance.update(script_sha256=sha256(__file__), python_version=platform.python_version(), numpy_version=np.__version__, uproot_version=uproot.__version__,
                      partition=partition, freeze_sha256=manifest["freeze_sha256"], production_run_uid=uid, production_binary_sha256=manifest["copied_binary_sha256"], production_seeds=manifest["seeds"], sampler_seed=args.seed,
                      sampler_rng_key="SHA256(seed|run_uid|raw_EventID) first16bytes interpreted as little-endian uint32 for numpy.default_rng")
    hit = {"schema_version": np.asarray(PREFIX+"_hit_truth_labels_v1"), "run_uid": np.asarray(uid), "partition_status": np.asarray(partition),
           "event_run_uid": np.full(ne, uid), "event_id_raw": np.arange(ne), "hit_row": np.arange(len(role)),
           "truth_class": role, "target_core": np.where(role.astype(np.int8) == 2, -1, role.astype(np.int8)).astype(np.int8),
           "microcurler_core_label": role == 1, "microcurler_family_label": role != 0,
           "root_sha256": np.asarray(provenance["root_sha256"]), "production_manifest_sha256": np.asarray(provenance["manifest_sha256"]),
           "truth_rule": np.asarray(TRUTH_RULE), **{k: root[k] for k in IDENTITIES}}
    electrons, etruth, events = export_electrons(root, role, ne, uid, args.seed, args.max_electrons, partition)
    binary, btruth = encode(electrons, etruth)
    for track in tracks:
        begin, end = electrons["event_ptr"][track["EventID"]:track["EventID"]+2]
        selected = etruth["track_id"][begin:end] == track["TrackID"]
        sites = np.unique(btruth["electron_to_site"][begin:end][selected])
        track["accepted_ideal_electrons"] = int(np.sum(selected))
        track["binary_sites_containing_track"] = len(sites)
        track["binary_supervised_core_sites_containing_track"] = int(np.sum(btruth["target_core"][sites] == 1))
        track["binary_ignored_sites_containing_track"] = int(np.sum(btruth["target_core"][sites] == -1))
    for e, row in enumerate(events):
        a, b = binary["event_ptr"][e:e+2]; target = btruth["target_core"][a:b]
        row["binary_supervised_other_core_ignored"] = [int(np.sum(target == c)) for c in (0, 1, -1)]
        row["binary_core_containing_sites"] = int(np.sum((btruth["role_bits"][a:b] & 2) != 0))
        row["binary_mixed_supervised_event"] = bool(np.any(target == 0) and np.any(target == 1))
        row["raw_core_without_accepted_core"] = row["source_core_rows"] > 0 and row["accepted_roles_other_core_descendant"][1] == 0
        row["raw_core_without_supervised_core_site"] = row["source_core_rows"] > 0 and not np.any(target == 1)
    paths = [out/name for name in OUTPUTS]
    save_npz(paths[0], hit)
    save_npz(paths[1], electrons)
    etruth.update(inputs_npz_sha256=np.asarray(sha256(paths[1])), hit_truth_labels_sha256=np.asarray(sha256(paths[0])), root_sha256=np.asarray(provenance["root_sha256"]))
    save_npz(paths[2], etruth)
    save_npz(paths[3], binary)
    btruth.update(inputs_npz_sha256=np.asarray(sha256(paths[3])), source_electron_inputs_sha256=np.asarray(sha256(paths[1])), source_electron_targets_sha256=np.asarray(sha256(paths[2])), root_sha256=np.asarray(provenance["root_sha256"]))
    save_npz(paths[4], btruth)
    write_csv(paths[5], events); write_csv(paths[6], tracks)
    scalar_sum = ("source_rows", "zero_charge_source_rows", "expected_electrons", "generated_electrons", "accepted_electrons", "lost_outside_square", "lost_disk_inside_square", "lost_outside_disk", "lost_square_boundary_inside_disk", "occupied_spacetime_sites", "duplicate_sites", "extra_tokens_at_duplicate_sites", "mixed_core_other_sites", "mixed_core_descendant_sites")
    totals = {k: sum(e[k] for e in events) for k in scalar_sum}
    for key in ("raw_roles_other_core_descendant", "generated_roles_other_core_descendant", "accepted_roles_other_core_descendant", "lost_roles_other_core_descendant", "binary_supervised_other_core_ignored", "accepted_electrons_negative_positive_end"):
        totals[key] = np.sum([e[key] for e in events], axis=0).tolist()
    totals.update(events=ne, raw_core_tracks=sum(t["truth_class"] == 1 for t in tracks), raw_core_events=sum(e["source_core_rows"] > 0 for e in events),
                  accepted_core_events=sum(e["accepted_roles_other_core_descendant"][1] > 0 for e in events),
                  binary_supervised_core_events=sum(e["binary_supervised_other_core_ignored"][1] > 0 for e in events),
                  binary_mixed_supervised_events=sum(e["binary_mixed_supervised_event"] for e in events),
                  empty_electron_binary_events=int(np.sum(np.diff(binary["event_ptr"]) == 0)),
                  empty_electron_binary_event_ids=np.flatnonzero(np.diff(binary["event_ptr"]) == 0).tolist(),
                  raw_core_tracks_with_accepted_electrons=sum(t["truth_class"] == 1 and t["accepted_ideal_electrons"] > 0 for t in tracks),
                  raw_core_tracks_with_supervised_core_sites=sum(t["truth_class"] == 1 and t["binary_supervised_core_sites_containing_track"] > 0 for t in tracks),
                  raw_core_tracks_without_accepted_electrons=[{"EventID":t["EventID"],"TrackID":t["TrackID"]} for t in tracks if t["truth_class"] == 1 and t["accepted_ideal_electrons"] == 0],
                  accepted_primary_other_electrons=int(np.sum((etruth["truth_class"] == 0) & (etruth["parent_id"] == 0))),
                  primary_other_bearing_binary_sites=int(np.sum(btruth["n_primary_other_electrons"] > 0)),
                  supervised_primary_other_binary_sites=int(np.sum((btruth["n_primary_other_electrons"] > 0) & (btruth["target_core"] == 0))),
                  ignored_primary_other_binary_sites=int(np.sum((btruth["n_primary_other_electrons"] > 0) & (btruth["target_core"] == -1))),
                  core_precedence_tracks=[{"EventID": t["EventID"], "TrackID": t["TrackID"]} for t in tracks if t["core_overrides_descendant"]],
                  raw_core_events_without_accepted_core=[e["event_id"] for e in events if e["raw_core_without_accepted_core"]],
                  raw_core_events_without_supervised_core_site=[e["event_id"] for e in events if e["raw_core_without_supervised_core_site"]])
    require(totals["accepted_electrons"]+sum(totals["lost_roles_other_core_descendant"]) == totals["generated_electrons"], "Electron acceptance conservation failed")
    audit = {"schema_version": PREFIX+"_readout_audit_v1", "status": "PASS", "partition_status": partition, "truth_rule": TRUTH_RULE,
             "provenance": provenance, "source_drift_contract_checks": checks, "totals": totals,
             "model_input_fields": ["global_pixel_x", "global_pixel_y", "readout_side", "time_bin_rel"],
             "metadata_not_model_features": ["event_ptr", "event_time_origin_bin_abs", "time_bin_abs", "all target-sidecar fields"],
             "assumptions": ELECTRON_ASSUMPTIONS+BINARY_ASSUMPTIONS, "events": events,
             "output_sha256": {p.name: sha256(p) for p in paths[:7]}}
    paths[7].write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    text = "FIMS EXPANDED TRANSPORT V2 TRUTH + IDEAL READOUT (V1 ENCODING)\nROOT / PRODUCTION / ANCESTRY / READOUT / ROUNDTRIP: PASS\n"+f"Run UID: {uid}\nProduction shard: {partition}; ML split is in the separate frozen event_split.csv; no model trained or scored.\n\n"+"\n".join(f"{k}: {v}" for k,v in totals.items())+"\n\nTRUTH RULE\n"+TRUTH_RULE+"\n\nASSUMPTIONS\n"+"\n".join(ELECTRON_ASSUMPTIONS+BINARY_ASSUMPTIONS)+"\n"
    paths[8].write_text(text); print(text.split("ASSUMPTIONS")[0], flush=True)
    for p in paths: print("Saved:", p, flush=True)


def self_test():
    import copy
    plan = {"schema": "fims_expanded_transport_protocol_v2", "source_sha256": production.TRIAL_SOURCE_HASHES,
            "transport": production.TRANSPORT_IDENTITY, "partitions": {name: {"events": spec["events"], "production_seeds": spec["seeds"],
                      "readout_seed": spec["readout_seed"]} for name, spec in PARTITION_SPECS.items()}}
    for part, expected in PARTITION_SPECS.items():
        spec = production.partition_spec(part, "a"*64, production.TRIAL_SOURCE_HASHES)
        uid = production.run_uid(spec)
        manifest = dict(spec, production_spec=spec.copy(), run_uid=uid, status="PASS", source_edits=True,
                        actual_source_edits=True, original_repository_edits=False, build_performed=False,
                        copied_binary_sha256=production.TRANSPORT_IDENTITY["binary_sha256"],
                        transport_environment=production.PRODUCTION_ENV, root_name=production.root_name(part))
        require(validate_production_manifest(manifest, plan) == (uid, 500), "V2 production contract failed")
        for key,value in (("events",1000),("seeds",[1,2]),("run_uid","old_uid"),("partition","wrong"),
                          ("root_name","../wrong.root"),("source_edits",False),("copied_binary_sha256","a"*64),
                          ("transport_environment",{})):
            bad=copy.deepcopy(manifest); bad[key]=value
            try: validate_production_manifest(bad,plan)
            except ValueError: pass
            else: raise AssertionError("Changed contract accepted: "+key)
        bad=copy.deepcopy(plan); bad["partitions"][part]["readout_seed"]+=1
        try: validate_production_manifest(manifest,bad)
        except ValueError: pass
        else: raise AssertionError("Changed sampler seed accepted")
    root = {k: np.zeros(5) for k in BRANCHES}
    root.update(EventID=np.zeros(5, np.int64), TrackID=np.array([1, 2, 3, 4, 5]), ParentID=np.array([0, 1, 2, 3, 1]),
                PDG=np.full(5, 11), MomIn=np.array([100., .2, .3, .03, .01]), HitPathLength_fromTrack=np.zeros(5),
                ParticleName=np.full(5, "e-"), CreatorProcess=np.array(["primary", "eIoni", "eIoni", "eIoni", "eIoni"]))
    role, tracks = label_tracks(root, 2, "test-run")
    require(role.tolist() == [0, 1, 1, 2, 0] and tracks[2]["core_overrides_descendant"], "Core precedence failed")
    bad = {k: v.copy() for k,v in root.items()}; bad["ParentID"][1] = 999
    try: label_tracks(bad, 2, "test-run")
    except ValueError: pass
    else: raise AssertionError("Missing ancestor accepted")
    bad["ParentID"][1] = 3
    try: label_tracks(bad, 2, "test-run")
    except ValueError: pass
    else: raise AssertionError("Ancestry cycle accepted")
    root["charge"] = np.array([1000., 1000., 1000., 1000., 0.])
    root["prePos_z"] = np.array([-50., -50., 50., 50., 0.]); root["postPos_z"] = root["prePos_z"].copy()
    distance = HALF_Z_MM-np.abs(root["postPos_z"])
    root["driftTime_notSmeared"] = distance/DRIFT_MM_NS
    root["sigmaT_mm"] = .30*np.sqrt(distance/10); root["sigmaL_mm"] = .20*np.sqrt(distance/10)
    x,y,a = export_electrons(root, role, 2, "run-A", 1729, 10000, "block00")
    x2,y2,a2 = export_electrons(root, role, 2, "run-A", 1729, 10000, "block00")
    require(all(np.array_equal(x[k],x2[k]) for k in x), "Per-run sampling is not deterministic")
    require(not np.array_equal(event_rng(1729,"run-A",0).random(5),event_rng(1729,"run-B",0).random(5)), "Run UID absent from RNG")
    require(y["target_core"].dtype == np.int8 and np.all(y["target_core"][y["truth_class"] == 2] == -1), "Unsigned target bug")
    # Partition changes alter target provenance only, never physical observables.
    xe, ye, ae = export_electrons(root, role, 2, "run-A", 1729, 10000, "block01")
    require(all(np.array_equal(x[k], xe[k]) for k in x) and str(ye["partition_status"]) == "block01",
            "Partition role leaked into physical sampling")
    bx,by = encode(x,y)
    be, te = encode(xe, ye)
    require(all(np.array_equal(bx[k], be[k]) for k in bx) and str(te["partition_status"]) == "block01",
            "Binary partition role differs or leaked into inputs")
    require(bx["event_ptr"][-1] == bx["event_ptr"][-2] and len(bx["event_ptr"]) == 3, "Empty event not retained")
    require(len(by["electron_to_site"]) == len(y["truth_class"]) and "event_split" not in by and "partition_status" not in bx, "Schema/split or conservation error")
    # Explicit all-role collision policy, preserving independent endcaps.
    token = {"truth_class": np.array([0,1,2,1,2,0,2],np.uint8), "readout_side": np.array([0,0,0,1,1,1,1]),
             "global_pixel_x": np.array([1,1,1,1,1,2,2]), "global_pixel_y": np.zeros(7), "time_bin_abs": np.zeros(7)}
    target,mask,amb,stats = make_targets_and_collision_mask(token)
    require(target.tolist()==[0,1,-1,1,-1,0,-1] and mask.tolist()==[False,False,False,True,False,True,False], "Collision/endcap policy failed")
    require(stats["mixed_core_other_sites"]==1, "Mixed-site count failed")
    print("SYNTHETIC SELF-TEST: PASS (ancestry, core precedence, uint8 signed targets, run-keyed RNG, both endcaps, collisions, empty events, twenty 500-event shard contracts, preassigned partition isolation)", flush=True)


# Reviewed pure sampler/aggregation functions and constants are inserted below.


IDENTITIES = ("EventID", "TrackID", "ParentID", "PDG")


BRANCHES = IDENTITIES + (
    "prePos_x", "prePos_y", "prePos_z", "postPos_x", "postPos_y", "postPos_z",
    "Gtime", "charge", "driftTime_notSmeared", "sigmaT_mm", "sigmaL_mm",
)


GAIN = 10.0


HALF_Z_MM = 50.0


RADIUS_MM = 50.0


DRIFT_MM_NS = 0.01


SIGMA_T_MM_SQRT_CM = 0.30


SIGMA_L_MM_SQRT_CM = 0.20


PIXEL_MM = 20.0 / 256.0


GLOBAL_PIXELS = 1280


CLOCK_NS = 25.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_array(value, name):
    arr = np.asarray(value)
    require(arr.ndim == 1, f"{name}: expected a one-dimensional array")
    require(np.all(np.isfinite(arr)), f"{name}: non-finite values")
    require(np.all(arr == np.floor(arr)), f"{name}: noninteger values")
    require(np.all(np.abs(arr.astype(np.float64)) < 2**53), f"{name}: integer outside supported exact range")
    return arr.astype(np.int64)


def validate_root_arrays(root, label):
    missing = sorted(set(BRANCHES) - set(root))
    require(not missing, f"{label}: missing ROOT branches {missing}")
    n = len(root["EventID"])
    result = {}
    for key in BRANCHES:
        arr = np.asarray(root[key])
        require(arr.ndim == 1 and len(arr) == n, f"{label}/{key}: row-shape mismatch")
        require(np.all(np.isfinite(arr)), f"{label}/{key}: non-finite values")
        result[key] = integer_array(arr, key) if key in IDENTITIES else arr.astype(np.float64)
    require(np.all(result["charge"] >= 0), f"{label}: negative charge")
    require(np.all(result["Gtime"] >= 0), f"{label}: negative Gtime")
    # Check the actual numerical contract, not just the presence of branch names.
    distance = np.maximum(0.0, HALF_Z_MM - np.abs(result["postPos_z"]))
    expected = {
        "driftTime_notSmeared": distance / DRIFT_MM_NS,
        "sigmaT_mm": SIGMA_T_MM_SQRT_CM * np.sqrt(distance / 10.0),
        "sigmaL_mm": SIGMA_L_MM_SQRT_CM * np.sqrt(distance / 10.0),
    }
    checks = {}
    for key, wanted in expected.items():
        residual = np.abs(result[key] - wanted)
        checks[key + "_max_abs_residual"] = float(residual.max(initial=0.0))
        require(np.allclose(result[key], wanted, rtol=2e-6, atol=2e-7),
                f"{label}: {key} does not match the current source model; max residual {residual.max(initial=0.0):.6g}. Stop and inspect the ROOT-generation provenance.")
    # The sampling model is only meaningful for step endpoints within the drift volume.
    for key in ("prePos_z", "postPos_z"):
        require(np.all(np.abs(result[key]) <= HALF_Z_MM + 1e-5),
                f"{label}: {key} extends outside +/-50 mm; inspect geometry before using this exporter")
    return result, checks


def event_rng(seed, sample, event_id):
    digest = hashlib.sha256(f"{seed}|{sample}|{int(event_id)}".encode("utf-8")).digest()
    return np.random.default_rng(np.frombuffer(digest[:16], dtype="<u4"))


def sample_event(root, role, row_indices, seed, sample, event_id, remaining_budget):
    """Sample one event; returned source rows are indices into the FULL ROOT tree."""
    rows = np.asarray(row_indices, dtype=np.int64)
    rng = event_rng(seed, sample, event_id)
    means = root["charge"][rows] / GAIN
    require(np.all(np.isfinite(means)) and np.all(means < 1e12), "Electron-count means are invalid or implausibly large")
    counts = rng.poisson(means)
    generated = sum(int(x) for x in counts)
    require(generated <= remaining_budget,
            f"Electron sampling budget exceeded at {sample} Event {event_id}: {generated} new electrons, {remaining_budget} budget remaining. No stream was truncated; increase --max-electrons deliberately if appropriate.")
    repeated = np.repeat(rows, counts)
    fraction = rng.random(generated)
    position = []
    for axis in "xyz":
        before = root[f"prePos_{axis}"][repeated]
        after = root[f"postPos_{axis}"][repeated]
        position.append(before + fraction * (after - before))
    x, y, z = position
    side = (z >= 0).astype(np.uint8)
    distance = np.maximum(0.0, HALF_Z_MM - np.abs(z))
    sigma_t = SIGMA_T_MM_SQRT_CM * np.sqrt(distance / 10.0)
    sigma_l = SIGMA_L_MM_SQRT_CM * np.sqrt(distance / 10.0)
    x = x + rng.normal(size=generated) * sigma_t
    y = y + rng.normal(size=generated) * sigma_t
    delay = np.maximum(0.0, distance + rng.normal(size=generated) * sigma_l) / DRIFT_MM_NS
    arrival = root["Gtime"][repeated] + delay
    require(np.all(np.isfinite(arrival)) and np.all(arrival / CLOCK_NS < np.iinfo(np.int64).max), "Invalid or overflowing arrival times")
    inside_disk = x * x + y * y <= RADIUS_MM**2
    inside_square = (x >= -RADIUS_MM) & (x < RADIUS_MM) & (y >= -RADIUS_MM) & (y < RADIUS_MM)
    accepted = inside_disk & inside_square
    px = np.floor((x[accepted] + RADIUS_MM) / PIXEL_MM).astype(np.int16)
    py = np.floor((y[accepted] + RADIUS_MM) / PIXEL_MM).astype(np.int16)
    require(np.all((px >= 0) & (px < GLOBAL_PIXELS) & (py >= 0) & (py < GLOBAL_PIXELS)), "Internal address mapping error")
    side = side[accepted]
    tb = np.floor(arrival[accepted] / CLOCK_NS).astype(np.int64)
    accepted_rows = repeated[accepted]
    accepted_role = role[accepted_rows]
    # Ordering uses only observed time/address, never truth. Equal sites retain draw order.
    order = np.lexsort((py, px, side, tb))
    token = {
        "global_pixel_x": px[order], "global_pixel_y": py[order],
        "readout_side": side[order], "time_bin_abs": tb[order],
        "source_hit_row": accepted_rows[order], "truth_class": accepted_role[order],
    }
    origin = int(tb.min()) if len(tb) else -1
    token["time_bin_rel"] = token["time_bin_abs"] - origin if len(tb) else np.empty(0, dtype=np.int64)
    represented = np.unique(accepted_rows)
    generated_roles = role[repeated]
    audit = {
        "sample": sample, "event_id": int(event_id), "source_rows": len(rows),
        "zero_charge_source_rows": int(np.count_nonzero(means == 0)),
        "expected_electrons": float(means.sum()), "generated_electrons": generated,
        "accepted_electrons": int(np.count_nonzero(accepted)),
        "lost_outside_square": int(np.count_nonzero(~inside_square)),
        "lost_disk_inside_square": int(np.count_nonzero(inside_square & ~inside_disk)),
        "lost_outside_disk": int(np.count_nonzero(~inside_disk)),
        "lost_square_boundary_inside_disk": int(np.count_nonzero(inside_disk & ~inside_square)),
        "generated_roles_other_core_descendant": np.bincount(generated_roles, minlength=3).tolist(),
        "accepted_roles_other_core_descendant": np.bincount(accepted_role, minlength=3).tolist(),
        "lost_roles_other_core_descendant": np.bincount(generated_roles[~accepted], minlength=3).tolist(),
        "source_rows_with_zero_sampled_electrons": int(np.count_nonzero(counts == 0)),
        "positive_charge_rows_with_zero_sampled_electrons": int(np.count_nonzero((means > 0) & (counts == 0))),
        "source_rows_with_generated_electrons_but_no_accepted_electrons": int(np.count_nonzero((counts > 0) & ~np.isin(rows, represented))),
        "source_rows_with_no_accepted_electrons": int(len(rows) - len(represented)),
        "source_core_rows": int(np.count_nonzero(role[rows] == 1)),
        "event_time_origin_bin_abs": origin,
        "accepted_electrons_negative_positive_end": np.bincount(side, minlength=2).tolist(),
    }
    return token, audit


def make_targets_and_collision_mask(token):
    role = np.asarray(token["truth_class"], dtype=np.uint8)
    # Convert to signed BEFORE assigning -1 (works with NumPy 2.5 unsigned rules).
    target = role.astype(np.int8, copy=True)
    target[role == 2] = -1
    if len(role):
        sites = np.column_stack([token[k] for k in ("readout_side", "global_pixel_x", "global_pixel_y", "time_bin_abs")])
        _, inverse, counts = np.unique(sites, axis=0, return_inverse=True, return_counts=True)
        has_core = np.bincount(inverse, weights=(role == 1), minlength=len(counts)) > 0
        has_other = np.bincount(inverse, weights=(role == 0), minlength=len(counts)) > 0
        has_desc = np.bincount(inverse, weights=(role == 2), minlength=len(counts)) > 0
        mixed = has_core & has_other
        ambiguous = mixed[inverse]
        audit = {
            "occupied_spacetime_sites": len(counts),
            "duplicate_sites": int(np.count_nonzero(counts > 1)),
            "extra_tokens_at_duplicate_sites": int(len(role) - len(counts)),
            "mixed_core_other_sites": int(np.count_nonzero(mixed)),
            "mixed_core_descendant_sites": int(np.count_nonzero(has_core & has_desc)),
            "ambiguous_site_tokens": int(np.count_nonzero(ambiguous)),
        }
    else:
        ambiguous = np.zeros(0, dtype=bool)
        audit = dict.fromkeys(("occupied_spacetime_sites", "duplicate_sites", "extra_tokens_at_duplicate_sites", "mixed_core_other_sites", "mixed_core_descendant_sites", "ambiguous_site_tokens"), 0)
    return target, (role != 2) & ~ambiguous, ambiguous, audit


def concatenate(parts, dtype):
    return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)


EXPECTED_SOURCE = {
    "src/ActionInitialization.cc": "6a0c58fa76bdd59ffeea63b7706fd8fb4a642b015bc2a659e21e987a86243263",
    "src/DetectorConstruction.cc": "5072edf15dc45ab5433218490993b0bdcbabc4e9b6ed99ee0946f631e304387e",
    "src/EventAction.cc": "4fc3a2317658f28b28a3562b030fabe3ff6b3b1ccf29eb3899a5feb82c35263f",
    "src/PrimaryGeneratorAction.cc": "6b56d888fdb3961210d362941ef61f1c441576d6aba4434d7c58aeebd4258982",
    "src/RunAction.cc": "99d131b40fbd787b8fc4e5406b71e5e0aa7b7bf52a1f9618cc210de0ff19d240",
    "src/SteppingAction.cc": "35f3ef5a5dd3863b41c8e57467138d3d2fae4087e2fe2f340df46255c8c05436",
    "src/TPCDigi.cc": "1b0ed2b76ba388b16f9413f2b3e262accba42920f7bfe0b5d392aa2ca0137a31",
    "src/TPCDigiCollection.cc": "fb230940f77de4b86c2cc8f0fc1282ee5c6ffe0734a640aa721a1c470a2fb8f9",
    "src/TPCDigitizer.cc": "f567e43a278e6d3f1c9df9e4eb62d227e8f56f044c4f40ab1c98d5f073251847",
    "src/TPCHit.cc": "50a950c15655fb0c5d3e4c713765fae37f030c77212eaf5b70ff4df5dc71016b",
    "src/TPCSensitiveDetector.cc": "555a3454d3d54037927d2627dcdedc4998a0189d1589f80098f746545fa0e8af",
    "include/ActionInitialization.hh": "a45b03180b8cd8625b4a8fd578ab562a8ce98616a04085f7d30e08d31b993a37",
    "include/DetectorConstruction.hh": "d0e971cf7eca4d47864a5a47fc0b9a0231370a42844ec7deaf074d9987c2abb2",
    "include/EventAction.hh": "b448ccf9fc150795d6dc1307418a07b3e0dd12a87aaab12730d3eb10989b075b",
    "include/PrimaryGeneratorAction.hh": "fa6dfb1b10e2749e4641e7a8365adbdfe5b6c45237e4ed0e94da3ad976cc6be4",
    "include/RunAction.hh": "d5fabac98809854004305e871ce0be6ff7dcc151f6cbd8f30f8e52a729aff541",
    "include/SteppingAction.hh": "37536f9ac73ac7b29a001cba174868de9f764ab7bd5f2c1fc957fb7f5e72348d",
    "include/TPCDigi.hh": "4ee22fa04dd651d7f91ad4c9cdd6db286d88407f460ad55e869abd44d3106cfb",
    "include/TPCDigiCollection.hh": "57ac7bb4a47b9cd9bbba869e12d44d117e74b0299ebc0a9458c207b051ee387e",
    "include/TPCDigitizer.hh": "e1c6585b0a32ad7d56c996f1546dcb52732e7a69c6cc7aee390b0d407ab17b8f",
    "include/TPCHit.hh": "c6979b32b8f0cd5ca133944cc84317424cbbd7d9fcc0868fbf02ff62fc36b8a5",
    "include/TPCSensitiveDetector.hh": "fe434e82cfed683cae35646ae99c8f1cf6786ac5fa09292a9e8108a4cb683a94",
}


ELECTRON_ASSUMPTIONS = ['Preliminary zero-noise sampled electron model; not a calibrated electronics or gas model.', 'ROOT charge is edep/(26 eV) multiplied by gain 10. Charge and readout formulas are unchanged; vacuum magnetic integration is explicitly versioned to tight100, with original gas manager retained. Production records actual source and copied binary hashes. The electron-count model remains idealized.', 'Per step N ~ Poisson(charge/10); Fano factor is 1 by construction, not a measured gas Fano factor. No explicit ionization-cluster correlations are modeled.', 'Each electron starts uniformly along the straight pre-to-post step segment; deposited energy is approximated as uniform along that segment.', 'All electrons in one step use its recorded post-step Gtime as an approximate creation time; exact per-electron production times are unavailable.', "Each electron drifts to the nearest z=-50 or z=+50 mm plane, matching the toy source's two-end convention. z=0 is assigned to the positive end.", 'Drift velocity is 0.01 mm/ns; sigmaT=0.30 sqrt(d/10) mm and sigmaL=0.20 sqrt(d/10) mm. Electron x, y and longitudinal Gaussian draws are independent.', 'Longitudinally smeared drift distance is clipped at zero; absolute arrival time is Gtime plus drift delay. Existing smeared step coordinates are not reused.', 'Only final xy arrivals within the radius-50 mm disk and enclosing [-50,50) mm square are accepted. No full drift-path wall-loss, attachment, gain fluctuation, efficiency or threshold model is included.', 'Each endcap has a 5x5 address grid of 20 mm chips, 256x256 pixels per chip (78.125 micrometre pitch), and 25 ns time bins. These are prototype discretization choices.', 'Each accepted ideal electron is one unit-amplitude token. Repeated same-address/same-time tokens are retained; real electronics may merge them. There is no hardware multi-hit resolution claim.', 'Truth roles and source identities are kept only in the target sidecar. Descendants are ignored for supervision. Tokens at an identical endcap/pixel/time site containing both core and other truth are also ignored for supervision, without changing their truth roles.', 'Event-relative time subtracts the first accepted arrival bin; empty events have origin -1. Time gaps remain intact. This is an offline event-delimited prototype, not a validated continuous readout pipeline.', 'All events, including empty events, are kept. partition_status identifies the production shard only; the separate frozen event_split.csv defines the ML train/validation assignment. No model is trained or scored here.']


BINARY_ASSUMPTIONS = ['One binary input spike per (event, readout end, global pixel x, global pixel y, 25 ns bin).', 'Coincident electrons are merged without exposing their number as an input feature.', 'No added noise, threshold inefficiency, or deadtime extending into another bin is modeled.', 'This is a prototype binary encoding, not a calibrated electronics response.', 'Event-relative time retains gaps. Event boundaries remain externally supplied in this offline prototype.', 'Descendant-only sites remain inputs but are ignored by the supervised loss.', 'Core plus descendant sites target core; other plus descendant sites target other.', 'Sites containing both core and other truth are ignored; no majority vote is used.', 'Truth composition, sample identity and multiplicity are diagnostics/targets, never model input features.', 'All events, including empty events, are kept. partition_status identifies the production shard only; the separate frozen event_split.csv defines the ML train/validation assignment. No model is trained or scored here.']


def encode(inputs, targets):
    validate_electrons(inputs, targets)
    old_ptr = inputs["event_ptr"]
    columns, counts_list, primary_list, ptr = [], [], [], [0]
    electron_to_site = np.empty(int(old_ptr[-1]), dtype=np.int64)
    for event, (begin, end) in enumerate(zip(old_ptr[:-1], old_ptr[1:])):
        begin, end = int(begin), int(end)
        if begin == end:
            ptr.append(ptr[-1])
            continue
        # Column order makes np.unique output time-first; truth never sets order.
        address = np.column_stack([inputs[key][begin:end] for key in
                                   ["time_bin_abs", "readout_side", "global_pixel_x", "global_pixel_y"]])
        unique, inverse = np.unique(address, axis=0, return_inverse=True)
        roles = targets["truth_class"][begin:end].astype(np.int64)
        counts = np.zeros((len(unique), 3), dtype=np.int64)
        np.add.at(counts, (inverse, roles), 1)
        primary = np.zeros(len(unique), dtype=np.int64)
        is_primary_other = (roles == 0) & (targets["parent_id"][begin:end] == 0)
        np.add.at(primary, inverse[is_primary_other], 1)
        conflict = (counts[:, 0] > 0) & (counts[:, 1] > 0)
        require(np.array_equal(targets["ambiguous_site_mask"][begin:end], conflict[inverse]), "Ambiguity mask disagrees with site truth")
        electron_to_site[begin:end] = inverse + ptr[-1]
        columns.append(unique)
        counts_list.append(counts)
        primary_list.append(primary)
        ptr.append(ptr[-1] + len(unique))
    sites = np.concatenate(columns) if columns else np.empty((0, 4), dtype=np.int64)
    counts = np.concatenate(counts_list) if counts_list else np.empty((0, 3), dtype=np.int64)
    primary = np.concatenate(primary_list) if primary_list else np.empty(0, dtype=np.int64)
    ptr = np.asarray(ptr, dtype=np.int64)
    index = np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), np.diff(ptr))
    bits = ((counts > 0).astype(np.uint8) * np.array([1, 2, 4], dtype=np.uint8)).sum(axis=1).astype(np.uint8)
    conflict = ((bits & 1) != 0) & ((bits & 2) != 0)
    target = np.full(len(sites), -1, dtype=np.int8)
    target[((bits & 2) != 0) & ~conflict] = 1
    target[((bits & 1) != 0) & ~conflict] = 0
    out_inputs = {
        "schema_version": np.asarray(PREFIX+"_binary_spike_inputs_v1"),
        "event_ptr": ptr,
        "event_time_origin_bin_abs": inputs["event_time_origin_bin_abs"].copy(),
        "global_pixel_x": sites[:, 2].astype(np.int16),
        "global_pixel_y": sites[:, 3].astype(np.int16),
        "readout_side": sites[:, 1].astype(np.uint8),
        "time_bin_abs": sites[:, 0],
        "time_bin_rel": sites[:, 0] - inputs["event_time_origin_bin_abs"][index],
        "pixel_pitch_mm": inputs["pixel_pitch_mm"].copy(),
        "clock_ns": inputs["clock_ns"].copy(),
        "grid_pixels_per_side": np.asarray(1280, dtype=np.int64),
        "readout_side_names": inputs["readout_side_names"].copy(),
        "model_assumptions_json": np.asarray(json.dumps(BINARY_ASSUMPTIONS)),
        "upstream_assumptions_json": inputs["model_assumptions_json"].copy(),
    }
    out_targets = {
        "schema_version": np.asarray(PREFIX+"_binary_spike_targets_v1"),
        "event_ptr": ptr.copy(), "site_event_index": index,
        "target_core": target, "supervision_mask": target >= 0,
        "role_bits": bits, "ambiguous_site_mask": conflict,
        "n_other_electrons": counts[:, 0], "n_core_electrons": counts[:, 1],
        "n_descendant_electrons": counts[:, 2], "n_primary_other_electrons": primary,
        "electron_to_site": electron_to_site,
    }
    for name in ["run_uid", "event_run_uid", "event_id_raw", "partition_status"]:
        out_targets[name] = targets[name].copy()
    require(int(counts.sum()) == int(old_ptr[-1]), "Electron conservation failure")
    for col, name in enumerate(["time_bin_abs", "readout_side", "global_pixel_x", "global_pixel_y"]):
        require(np.array_equal(sites[electron_to_site, col], inputs[name]), f"Roundtrip coordinate failure: {name}")
    require(np.array_equal(index[electron_to_site], targets["hit_event_index"]), "Roundtrip event failure")
    return out_inputs, out_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--freeze-file", type=Path)
    parser.add_argument("--max-electrons", type=int, default=50_000_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test()
    elif args.production_dir is None or args.freeze_file is None or args.seed is None:
        parser.error("Provide --production-dir, --freeze-file and --seed")
    else: main_run(args)
