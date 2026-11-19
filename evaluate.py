"""Run the course robustness, thermal restoration or frozen residual evaluation."""
import argparse
from collections import Counter
import csv
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from degradations import CONDITIONS, FINAL_CONDITIONS, METHODS
from model import (FAMILIES, ROOT, FeatureTap, ModelRuntime, add_runtime_arguments, final_logits,
                   load_bundle, logits_numpy, prepare_inputs, read_json, resolve_assets, scene_of,
                   sha256_file, split_ids, verify_split_roles, write_json)
from reliability import (MetricsAccumulator, fit_sample_indices, fit_temperature, metric_rows,
                         paired_transitions, probability_at_temperature, sample_indices,
                         segmentation, summarize_extra)

SCHEMES = ("T1", "TS_V", "TS_F", "TS_DN_F")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if not isinstance(v, float) or np.isfinite(v) else ""
                             for k, v in row.items()})


def temperature_for(table, family, scheme, scene):
    if scheme == "T1":
        return 1.0
    value = table[family][scheme]
    if isinstance(value, dict):
        value = value[scene]
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid temperature for {family}/{scheme}/{scene}.")
    return value


def fit_clean_temperatures(runtime, ids, forward, families, settings, include_scene=False):
    """Fit scalar temperatures from fixed CLEAN development pixels."""
    parts = {m: [] for m in families}
    labels, weights, scene_parts = [], [], []
    for image_id in ids:
        rgb, thermal, y = runtime.read("development", image_id)
        indices = fit_sample_indices(y, settings["evaluation_seed"], image_id,
                                     settings["calibration_pixels_per_image"])
        outputs = forward(rgb, thermal, y, image_id, CONDITIONS["CLEAN"])
        for m in families:
            parts[m].append(logits_numpy(outputs[m]).reshape(-1, 9)[indices])
        labels.append(y.ravel()[indices])
        valid_count = int(np.count_nonzero((0 <= y) & (y < 9)))
        weights.append(np.full(len(indices), valid_count / len(indices)))
        scene_parts.append(np.full(len(indices), scene_of(image_id), dtype="U1"))

    y, w, scenes = map(np.concatenate, (labels, weights, scene_parts))
    table, fits = {}, {}
    for m in families:
        z = np.concatenate(parts[m])
        fits[m] = {group: fit_temperature(z, y, w, group) for group in ("V", "F")}
        table[m] = {"T1": 1., **{f"TS_{g}": fit["temperature"] for g, fit in fits[m].items()}}
        if include_scene:
            for scene in ("D", "N"):
                mask = scenes == scene
                fits[m][f"{scene}_F"] = fit_temperature(z[mask], y[mask], w[mask], "F")
            table[m]["TS_DN_F"] = {scene: fits[m][f"{scene}_F"]["temperature"] for scene in ("D", "N")}
    return {"temperatures": table, "fits": fits, "fit_ids": list(ids),
            "fit_condition": "CLEAN", "test_used_for_fit": False}


def merge_accumulators(accumulators):
    combined = MetricsAccumulator(accumulators[0].classes, next(iter(accumulators[0].groups.values())).bins)
    for accumulator in accumulators:
        combined.confusion += accumulator.confusion
        combined.samples.extend(accumulator.samples)
        for group, source in accumulator.groups.items():
            dest = combined.groups[group]
            for field in ("count", "correct", "conf_sum", "nll_sum", "n", "error_count"):
                setattr(dest, field, getattr(dest, field) + getattr(source, field))
            for score in dest.hist:
                dest.hist[score] += source.hist[score]
    return combined


def _metric_row(split, cid, family, scheme, temperature, acc, scene=None):
    result = summarize_extra(acc.finish())
    seg = segmentation(acc.confusion)
    row = metric_rows(split, cid, scheme, temperature, result, len(acc.samples))
    row.update(family=family, background_false_positive_fraction=seg["background_false_positive_fraction"],
               foreground_binary_precision=seg["foreground_binary_precision"])
    if scene is not None:
        row["scene"] = scene
    return row


def _probabilities(runtime, tensor, temperature, use_original_fp32, array=None):
    if use_original_fp32:
        return (logits_numpy(runtime.torch.softmax(tensor.float(), dim=1)),
                logits_numpy(runtime.torch.log_softmax(tensor.float(), dim=1)))
    return probability_at_temperature(logits_numpy(tensor) if array is None else array, temperature)


def evaluate(runtime, settings, split, ids, conditions, families, schemes, temperatures, forward, output,
             fp32_t1=False):
    summary, conditional, pooled, class_rows, transitions = [], [], [], [], []
    scenes = tuple(dict.fromkeys(map(scene_of, ids)))
    comparisons = tuple(combinations(families, 2))
    for cid in conditions:
        acc = {(m, scheme, scene): MetricsAccumulator(9, settings["ece_bins"])
               for m in families for scheme in schemes for scene in scenes}
        pair_counts = {pair: Counter() for pair in comparisons}
        for index, image_id in enumerate(ids, 1):
            rgb, thermal, y = runtime.read(split, image_id)
            scene = scene_of(image_id)
            outputs = forward(rgb, thermal, y, image_id, CONDITIONS[cid])
            indices = sample_indices(y, settings["auroc_pixels_per_image"], settings["evaluation_seed"], image_id)
            predictions = {}
            for family, tensor in outputs.items():
                raw = logits_numpy(tensor)
                for scheme in schemes:
                    t = temperature_for(temperatures, family, scheme, scene)
                    p, logp = _probabilities(runtime, tensor, t, fp32_t1 and scheme == "T1", raw)
                    maps = acc[family, scheme, scene].update(p, y, indices, image_id, logp)
                    pred = maps["prediction"]
                    if family in predictions and not np.array_equal(predictions[family], pred):
                        raise ValueError(f"Temperature changed predicted labels for {family}/{image_id}.")
                    predictions[family] = pred
            for first, second in comparisons:
                pair_counts[first, second].update(paired_transitions(y, predictions[first], predictions[second]))
            if index == 1 or index % 25 == 0 or index == len(ids):
                print(f"[{split} {cid}] {index}/{len(ids)}", flush=True)

        for family in families:
            for scheme_index, scheme in enumerate(schemes):
                parts = [acc.pop((family, scheme, scene)) for scene in scenes]
                combined = merge_accumulators(parts)
                t = temperature_for(temperatures, family, scheme, scenes[0]) if scheme != "TS_DN_F" else "D/N"
                summary.append(_metric_row(split, cid, family, scheme, t, combined))
                for scene, part in zip(scenes, parts):
                    t = temperature_for(temperatures, family, scheme, scene)
                    conditional.append(_metric_row(split, cid, family, scheme, t, part, scene))
                for group in ("V", "F"):
                    n = combined.groups[group].n
                    gaps = [item.groups[group].correct - item.groups[group].conf_sum for item in parts]
                    joint = float(np.abs(np.sum(gaps, axis=0)).sum() / n) if n else None
                    weighted = float(sum(np.abs(g).sum() for g in gaps) / n) if n else None
                    pooled.append({"condition": cid, "family": family, "scheme": scheme, "group": group,
                                   "pooled_ECE": joint, "pixel_weighted_scene_ECE": weighted,
                                   "cancellation_gap": weighted - joint if n else None})
                if scheme_index == 0:
                    for scene, part in zip(("all", *scenes), (combined, *parts)):
                        q = part.confusion
                        ious = segmentation(q)["class_IoU_pct"]
                        for i, name in enumerate(settings["class_names"]):
                            class_rows.append({"condition": cid, "family": family, "scene": scene,
                                               "class_id": i, "class_name": name, "IoU_pct": float(ious[i]),
                                               "GT_pixels": int(q[i].sum()),
                                               "predicted_pixels": int(q[:, i].sum())})
        for (first, second), counts in pair_counts.items():
            transitions.append({"condition": cid, "reference": first, "candidate": second, **dict(counts)})
        # Checkpoint of completed conditions, useful for long GPU runs.
        for file, rows in (("summary.csv", summary), ("conditional_summary.csv", conditional),
                           ("pooled_vs_conditional_ECE.csv", pooled), ("per_class_IoU.csv", class_rows),
                           ("paired_transitions.csv", transitions)):
            write_csv(output / file, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--preset", choices=("robustness", "restoration", "final"), default="final")
    parser.add_argument("--split", choices=("val", "development", "val_diagnostic", "test"), default="test")
    parser.add_argument("--condition", choices=tuple(CONDITIONS), action="append",
                        help="Repeat to override the preset's condition list.")
    parser.add_argument("--scheme", choices=SCHEMES, action="append", help="Repeat to select probability schemes.")
    parser.add_argument("--bundle", type=Path, help="Residual bundle produced by train_residual.py.")
    parser.add_argument("--temperatures", type=Path, help="Temperatures fitted for the specified residual bundle.")
    parser.add_argument("--fit-clean", action="store_true", help="Fit requested temperatures on development CLEAN.")
    parser.add_argument("--limit", type=int, help="Quick evaluation subset. Calibration still uses development.")
    parser.add_argument("--output", required=True, type=Path, help="New directory for evaluation CSV files.")
    args = parser.parse_args(argv)
    if args.bundle and args.preset != "final":
        parser.error("--bundle applies to the final preset")
    if args.temperatures and args.fit_clean:
        parser.error("Use one of --temperatures or --fit-clean")
    if args.fit_clean and args.split in ("val", "development"):
        parser.error("CLEAN fits must be evaluated on val_diagnostic or test")
    families = {"robustness": ("RAW",), "restoration": METHODS, "final": FAMILIES}[args.preset]
    conditions = list(dict.fromkeys(args.condition or (
        tuple(CONDITIONS) if args.preset == "robustness" else FINAL_CONDITIONS)))
    schemes = list(dict.fromkeys(args.scheme or (SCHEMES[:3] if args.preset == "final" else ("T1",))))
    fitted_schemes = set(schemes) - {"T1"}
    if fitted_schemes and args.split in ("val", "development"):
        parser.error("Use val_diagnostic or test to evaluate fitted temperatures")
    if args.bundle and fitted_schemes and not (args.temperatures or args.fit_clean):
        parser.error("A custom residual bundle needs matching --temperatures or --fit-clean for calibrated schemes")
    if (args.preset == "final" and "TS_DN_F" in schemes and not (args.temperatures or args.fit_clean)):
        parser.error("Final frozen temperatures cover T1, TS_V and TS_F. To fit TS_DN_F use --fit-clean.")
    verify_split_roles()
    settings = resolve_assets(args)
    ids = split_ids(args.split, args.limit)
    runtime = ModelRuntime(settings)
    tap = None
    try:
        if args.preset == "final":
            bundle = args.bundle or ROOT / "assets" / "frozen_residual_bundle.npz"
            bundle_hash = sha256_file(bundle)
            if not args.bundle and bundle_hash != settings["frozen_asset_sha256"][bundle.name]:
                raise ValueError("Frozen residual asset failed SHA-256 verification.")
            heads, choices = load_bundle(bundle, runtime.device)
            tap = FeatureTap(runtime)

            def forward(rgb, thermal, y, image_id, condition):
                rgb, thermal = prepare_inputs(rgb, thermal, condition, image_id, settings["evaluation_seed"], "GAUSS_0P8")
                return final_logits(runtime, tap, heads, choices, rgb, thermal, y, scene_of(image_id))
        else:
            bundle_hash, choices = None, None

            def forward(rgb, thermal, y, image_id, condition):
                outputs = {}
                for family in families:
                    r, t = prepare_inputs(rgb, thermal, condition, image_id, settings["evaluation_seed"], family)
                    outputs[family] = runtime.forward(r, t)
                return outputs

        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        config = {"preset": args.preset, "split": args.split, "subset": args.limit is not None, "image_ids": ids,
                  "conditions": conditions, "schemes": schemes, "checkpoint_sha256": runtime.checkpoint_hash,
                  "bundle_sha256": bundle_hash, "residual_choices": choices}
        write_json(output / "evaluation_config.json", config)
        if args.fit_clean:
            fitted = fit_clean_temperatures(runtime, split_ids("development"), forward, families, settings,
                                            include_scene="TS_DN_F" in schemes)
            fitted.update(base_checkpoint_sha256=runtime.checkpoint_hash, residual_bundle_sha256=bundle_hash)
        elif fitted_schemes or args.temperatures:
            path = args.temperatures or ROOT / "assets" / (
                "frozen_temperatures.json" if args.preset == "final" else "restoration_temperatures.json")
            if path.name in settings["frozen_asset_sha256"] and not args.temperatures:
                if sha256_file(path) != settings["frozen_asset_sha256"][path.name]:
                    raise ValueError("Frozen temperature asset failed SHA-256 verification.")
            fitted = read_json(path)
            if args.bundle and fitted.get("residual_bundle_sha256") != bundle_hash:
                raise ValueError("The supplied temperatures are not tied to this residual bundle.")
            if set(fitted["fit_ids"]) != set(split_ids("development")) or fitted.get("fit_condition") != "CLEAN":
                raise ValueError("Temperatures must come from the fixed development CLEAN split.")
        else:
            fitted = {"temperatures": {family: {"T1": 1.0} for family in families}}
        temperatures = fitted["temperatures"]
        for m in families:
            for scheme in schemes:
                for scene in ("D", "N"):
                    temperature_for(temperatures, m, scheme, scene)
        write_json(output / "temperatures.json", fitted)
        evaluate(runtime, settings, args.split, ids, conditions, families, schemes, temperatures, forward, output,
                 fp32_t1=args.preset == "robustness" and not fitted_schemes)
        print(f"Evaluation complete: {output}")
    finally:
        if tap is not None:
            tap.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Evaluation failed: {error}")
