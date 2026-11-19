"""Train B and C on frozen SwiftFormer-CMX+ features, then select and calibrate."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

from degradations import CONDITIONS, FINAL_CONDITIONS, restore_thermal
from evaluate import fit_clean_temperatures, write_csv
from model import (BUNDLE_FORMAT, EPOCHS, FAMILIES, GAMMAS, FeatureTap, ModelRuntime,
                   add_runtime_arguments, final_logits, head_arrays, load_heads, new_heads,
                   prepare_inputs, residual_logits, resolve_assets, scene_of, sha256_file,
                   split_ids, verify_split_roles, write_json)
from reliability import segmentation


def _seed(seed, *fields):
    key = "|".join(map(str, (seed, *fields))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def epoch_plan(ids, epoch, seed):
    """Keep the original RNG namespaces so the course augmentation is reproducible."""
    order = np.random.default_rng(_seed(seed, "batch11_image_order", epoch)).permutation(len(ids))
    for i in order:
        image_id = ids[i]
        rng = np.random.default_rng(_seed(seed, "batch11_augmentation", epoch, image_id))
        sigma = 0. if rng.random() < .5 else float(rng.choice([.05, .10]))
        yield image_id, sigma


def training_noise(thermal, image_id, epoch, sigma, seed):
    a = np.asarray(thermal, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not all(
        np.array_equal(a[..., 0], a[..., i]) for i in (1, 2)
    ):
        raise ValueError("Expected repeated grayscale channels in the training thermal image.")
    if sigma == 0:
        return a.copy()
    rng = np.random.default_rng(_seed(seed, "batch11_training_noise", epoch, image_id))
    noisy = np.clip(a[..., 0].astype(float) + rng.normal(0, 255 * sigma, a.shape[:2]),
                   0, 255).astype(np.float32)
    return np.repeat(noisy[..., None], 3, axis=-1)


def network_digest(network):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_frozen(network):
    if any(p.requires_grad or p.grad is not None for p in network.parameters()):
        raise RuntimeError("The shared model acquired trainable parameters or gradients.")
    if any(module.training for module in network.modules()):
        raise RuntimeError("The shared model left evaluation mode.")


def train_heads(runtime, tap, heads, ids, checkpoints, output, settings):
    torch = runtime.torch
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-4, weight_decay=1e-4,
                                  betas=(.9, .999), eps=1e-8)
    seed = settings["training_seed"]
    curves, snapshots = [], {}
    state = network_digest(runtime.network)
    folder = output / "checkpoints"
    folder.mkdir()
    for epoch in range(1, max(checkpoints) + 1):
        heads.train()
        total = {"B": 0., "C": 0.}
        for step, (image_id, sigma) in enumerate(epoch_plan(ids, epoch, seed), 1):
            rgb, thermal, label = runtime.read("train", image_id)
            thermal = restore_thermal(training_noise(thermal, image_id, epoch, sigma, seed), "GAUSS_0P8")[0]
            features, low, _ = runtime.tapped_forward(tap, rgb, thermal)
            scene = scene_of(image_id)
            target = torch.from_numpy(label.copy()).to(device=runtime.device, dtype=torch.long)[None]
            optimizer.zero_grad(set_to_none=True)
            losses = {
                family: torch.nn.functional.cross_entropy(
                    residual_logits(heads, features, low, family, scene, label.shape, 1.),
                    target, ignore_index=255)
                for family in ("B", "C")
            }
            loss = losses["B"] + losses["C"]
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite paired loss at epoch {epoch}, image {image_id}.")
            loss.backward()
            inactive = "N" if scene == "D" else "D"
            if any(p.grad is not None for p in heads.experts[inactive].parameters()):
                raise RuntimeError("The inactive scene head received a gradient.")
            active = (*heads.unified.parameters(), *heads.experts[scene].parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in active):
                raise RuntimeError("The selected heads did not receive finite gradients.")
            optimizer.step()
            for family, value in losses.items():
                total[family] += float(value.item())
            if step == 1 or step % 100 == 0 or step == len(ids):
                print(f"[Train {epoch:02d}/{max(checkpoints)}] {step}/{len(ids)}", flush=True)
        check_frozen(runtime.network)
        curves.append({"epoch": epoch, "images": len(ids), "mean_loss_B": total["B"] / len(ids),
                       "mean_loss_C": total["C"] / len(ids)})
        write_csv(output / "training_curve.csv", curves)
        if epoch in checkpoints:
            snapshots[epoch] = head_arrays(heads)
            np.savez_compressed(folder / f"epoch_{epoch:03d}.npz", **snapshots[epoch])
            if network_digest(runtime.network) != state:
                raise RuntimeError("Frozen model tensors changed during residual training.")
    return snapshots


def confusion(label, prediction):
    valid = (0 <= label) & (label < 9)
    return np.bincount(9 * label[valid].astype(np.int64) + prediction[valid],
                       minlength=81).reshape(9, 9)


def select_family(family, means):
    """Choose an epoch and one gamma per family on development only."""
    base = means["A", 0, 0.]
    clean_base = base["CLEAN"]["mIoU9_pct"]
    base_score = (base["T-NOISE-L2"]["mIoU9_pct"] + base["T-NOISE-L3"]["mIoU9_pct"]) / 2
    zero = {"epoch": 0, "gamma": 0., "score": base_score, "clean": clean_base, "eligible": True,
            "background_FP_increase_pp": {cid: 0. for cid in FINAL_CONDITIONS}}
    rows = [zero]
    for (m, epoch, gamma), record in means.items():
        if m != family:
            continue
        clean = record["CLEAN"]["mIoU9_pct"]
        fp = {cid: 100 * (record[cid]["background_false_positive_fraction"]
                        - base[cid]["background_false_positive_fraction"]) for cid in FINAL_CONDITIONS}
        score = (record["T-NOISE-L2"]["mIoU9_pct"] + record["T-NOISE-L3"]["mIoU9_pct"]) / 2
        rows.append({"epoch": epoch, "gamma": gamma, "score": score, "clean": clean,
                     "background_FP_increase_pp": fp,
                     "eligible": bool(clean >= clean_base - 1. - 1e-10
                                      and all(v <= .2 + 1e-10 for v in fp.values()))})
    best = max((r for r in rows if r["eligible"]),
               key=lambda r: (r["score"], r["clean"], -r["epoch"], -r["gamma"]))
    if best["score"] <= base_score + 1e-8:
        best = zero
    return best, rows


def scan_development(runtime, tap, snapshots, ids, settings):
    bank = {epoch: load_heads(new_heads().to(runtime.device), arrays, runtime.device).eval()
            for epoch, arrays in snapshots.items()}
    baseline = ("A", 0, 0.)
    grid = [(m, epoch, gamma) for m in ("B", "C") for epoch in sorted(bank)
            for gamma in GAMMAS if gamma > 0]
    means = {key: {} for key in (baseline, *grid)}
    for cid in FINAL_CONDITIONS:
        matrices = {key: np.zeros((9, 9), dtype=np.int64) for key in means}
        for index, image_id in enumerate(ids, 1):
            rgb, thermal, y = runtime.read("development", image_id)
            rgb, thermal = prepare_inputs(rgb, thermal, CONDITIONS[cid], image_id,
                                          settings["evaluation_seed"], "GAUSS_0P8")
            f, low, native = runtime.tapped_forward(tap, rgb, thermal)
            matrices[baseline] += confusion(y, native.argmax(1)[0].cpu().numpy())
            with runtime.torch.no_grad():
                for key in grid:
                    family, epoch, gamma = key
                    z = residual_logits(bank[epoch], f, low, family, scene_of(image_id), y.shape, gamma)
                    matrices[key] += confusion(y, z.argmax(1)[0].cpu().numpy())
            if index == 1 or index % 25 == 0 or index == len(ids):
                print(f"[Development {cid}] {index}/{len(ids)}", flush=True)
        for key, q in matrices.items():
            means[key][cid] = segmentation(q)
    return means


def save_selection(output, snapshots, decisions, device):
    chosen = head_arrays(new_heads())
    for key in chosen:
        family = "B" if key.startswith("unified.") else "C"
        epoch = decisions[family]["epoch"]
        if epoch:
            chosen[key] = snapshots[epoch][key]
    choices = {m: (d["epoch"], d["gamma"]) for m, d in decisions.items()}
    info = {f"{m}_{k}": np.array(d[k]) for m, d in decisions.items() for k in ("epoch", "gamma")}
    bundle = output / "residual_bundle.npz"
    np.savez_compressed(bundle, format=np.array(BUNDLE_FORMAT), **info,
                        **{"head__" + k: v for k, v in chosen.items()})
    heads = load_heads(new_heads().to(device), chosen, device).requires_grad_(False).eval()
    return bundle, heads, choices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="One training epoch on two images; writes trial heads only.")
    args = parser.parse_args(argv)
    verify_split_roles()
    settings = resolve_assets(args)
    ids = split_ids("train", 2 if args.smoke else None)
    development_ids = split_ids("development")
    runtime = ModelRuntime(settings)
    tap = None
    try:
        tap = FeatureTap(runtime)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "training_config.json",
                   {"checkpoint_sha256": runtime.checkpoint_hash, "training_ids": ids,
                    "seed": settings["training_seed"], "smoke": args.smoke,
                    "epochs": 1 if args.smoke else 20, "learning_rate": 1e-4, "weight_decay": 1e-4})
        heads = new_heads().to(runtime.device)
        # Verify that the hook and interpolation reconstruct the underlying model logits.
        image_id = ids[0]
        rgb, thermal, y = runtime.read("train", image_id)
        rgb, thermal = prepare_inputs(rgb, thermal, CONDITIONS["CLEAN"], image_id,
                                      settings["evaluation_seed"], "GAUSS_0P8")
        f, low, native = runtime.tapped_forward(tap, rgb, thermal)
        with runtime.torch.no_grad():
            for m in ("B", "C"):
                zero = residual_logits(heads, f, low, m, scene_of(image_id), y.shape, 1.)
                if not runtime.torch.allclose(zero, native, rtol=0, atol=1e-5):
                    raise RuntimeError("Zero residual does not reconstruct the native logits.")
        del f, low, native, zero
        snapshots = train_heads(runtime, tap, heads, ids, (1,) if args.smoke else EPOCHS, output, settings)
        if args.smoke:
            print(f"Training smoke check complete: {output}")
            return 0
        means = scan_development(runtime, tap, snapshots, development_ids, settings)
        decisions, audit = {}, {}
        for family in ("B", "C"):
            decisions[family], audit[family] = select_family(family, means)
        write_json(output / "selection.json", {"selected": decisions, "grid": audit,
                                               "split": "development", "test_used_for_selection": False})
        bundle, selected, choices = save_selection(output, snapshots, decisions, runtime.device)

        def forward(rgb, thermal, y, image_id, condition):
            rgb, thermal = prepare_inputs(rgb, thermal, condition, image_id,
                                          settings["evaluation_seed"], "GAUSS_0P8")
            return final_logits(runtime, tap, selected, choices, rgb, thermal, y, scene_of(image_id))

        temps = fit_clean_temperatures(runtime, development_ids, forward, FAMILIES, settings)
        temps.update(base_checkpoint_sha256=runtime.checkpoint_hash, residual_bundle_sha256=sha256_file(bundle))
        write_json(output / "temperatures.json", temps)
        print(f"Training, development selection and calibration complete: {output}")
    finally:
        if tap is not None:
            tap.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Training failed: {error}")
