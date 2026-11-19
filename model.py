"""Bridge to the original SwiftFormer-CMX+ model and its MFNet input pipeline."""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np

from degradations import apply_corruption, restore_thermal

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "course.json"
BUNDLE_FORMAT = "Batch12_unscaled_weights_gamma_on_residual_output_v1"
GAMMAS = (0., .25, .5, .75, 1.)
EPOCHS = (5, 10, 15, 20)
FAMILIES = ("A", "B", "C")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(_plain(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_of(image_id):
    if not re.fullmatch(r"[0-9]+[DN]", image_id):
        raise ValueError(f"Expected a fixed MFNet ID ending in D or N: {image_id!r}")
    return image_id[-1]


def split_path(split):
    if split not in ("train", "val", "development", "val_diagnostic", "test"):
        raise ValueError(f"Unknown split {split!r}")
    return ROOT / "configs" / "splits" / f"{split}.txt"


def split_ids(split, limit=None):
    path = split_path(split)
    ids = path.read_text(encoding="utf-8-sig").split()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate split IDs: {path}")
    for image_id in ids:
        scene_of(image_id)
    if limit is not None:
        if not 1 <= limit <= len(ids):
            raise ValueError(f"--limit must be between 1 and {len(ids)} for {split}")
        # Retain both scenes in a quick run whenever the limit permits it.
        if limit >= 2:
            first = {next(i for i in ids if scene_of(i) == s) for s in ("D", "N")}
            first.update([i for i in ids if i not in first][:limit - 2])
            return [i for i in ids if i in first]
        return ids[:limit]
    return ids


def verify_split_roles():
    ids = {k: set(split_ids(k)) for k in ("train", "val", "test", "development", "val_diagnostic")}
    if ids["development"] & ids["val_diagnostic"] or ids["development"] | ids["val_diagnostic"] != ids["val"]:
        raise ValueError("The development and diagnostic lists must partition validation.")
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise ValueError("Training, validation and test lists must be disjoint.")


def add_runtime_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path, help="Original model project root (models/, dataloader/, engine/).")
    parser.add_argument("--dataset-root", type=Path, help="Prepared MFNet root with images/, Thermal/ and labels/.")
    parser.add_argument("--checkpoint", type=Path, help="Course SwiftFormer-CMX+ base checkpoint.")
    parser.add_argument("--device", help="PyTorch device, e.g. cuda:0.")


def resolve_assets(args):
    settings = read_json(args.config)
    for key in ("model_root", "dataset_root", "checkpoint"):
        override = getattr(args, key)
        settings[key] = override.expanduser().resolve() if override is not None else (
            ROOT / Path(settings[key]).expanduser()).resolve()
    if args.device:
        settings["device"] = args.device
    return settings


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _verified_project(settings):
    project = settings["model_root"]
    expected = settings["model_source_sha256"]
    missing = [key for key in expected if not (project / key).is_file()]
    if missing:
        raise FileNotFoundError(f"Original SwiftFormer-CMX+ implementation missing at {project}: "
                                + ", ".join(missing) + ". Supply --model-root.")
    changed = [key for key, checksum in expected.items() if sha256_file(project / key) != checksum]
    if changed:
        raise ValueError("Model source revision differs from the archived course revision: "
                         + ", ".join(changed))
    return project


def _load_checkpoint(network, checkpoint, torch):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("The checkpoint must contain a model state_dict.")
    state = next((payload[k] for k in ("model", "state_dict", "module")
                  if isinstance(payload.get(k), Mapping)), payload)
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("criterion."):
            continue
        if key in cleaned or not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid model checkpoint entry: {key}")
        cleaned[key] = value
    for key, value in network.state_dict().items():
        if key.endswith("num_batches_tracked") and key not in cleaned:
            cleaned[key] = value
    network.load_state_dict(cleaned, strict=True)


class ModelRuntime:
    def __init__(self, settings):
        project = _verified_project(settings)
        data = settings["dataset_root"]
        for folder in ("images", "Thermal", "labels"):
            if not (data / folder).is_dir():
                raise FileNotFoundError(f"Missing prepared MFNet folder: {data / folder}. Supply --dataset-root.")
        checkpoint = settings["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Base checkpoint missing: {checkpoint}. Supply --checkpoint.")
        self.checkpoint_hash = sha256_file(checkpoint)
        if self.checkpoint_hash != settings["checkpoint_sha256"]:
            raise ValueError("Base checkpoint failed SHA-256 verification.")
        import torch
        self.torch = torch
        self.device = torch.device(settings["device"])
        self.image_size = tuple(settings["image_size"])
        self.datasets = {}
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Install the CUDA build of PyTorch for the course setup.")
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
        seed = int(settings["evaluation_seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_float32_matmul_precision("high")

        sys.path.insert(0, str(project))
        with working_directory(project):
            spec = importlib.util.spec_from_file_location("config", ROOT / "configs" / "mfnet_base.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["config"] = module
            spec.loader.exec_module(module)
            cfg = module.config
            cfg.root_dir = cfg.abs_dir = str(project)
            cfg.dataset_path = str(data)
            cfg.rgb_root_folder, cfg.x_root_folder, cfg.gt_root_folder = (
                str(data / folder) for folder in ("images", "Thermal", "labels"))
            cfg.train_source = str(split_path("train"))
            cfg.eval_source = str(split_path("development"))
            cfg.num_train_imgs = len(split_ids("train"))
            cfg.num_eval_imgs = len(split_ids("development"))
            if (cfg.num_classes != 9 or cfg.gt_transform or list(cfg.class_names) != settings["class_names"]
                    or not np.array_equal(cfg.norm_mean, settings["mean"])
                    or not np.array_equal(cfg.norm_std, settings["std"])):
                raise ValueError("The MFNet label convention or normalization differs from the course config.")
            from models.builder import EncoderDecoder
            from dataloader.RGBXDataset import RGBXDataset
            from dataloader.dataloader import ValPre
            from engine.evaluator import Evaluator
            self.dataset_class, self.preprocess_class = RGBXDataset, ValPre
            self.network = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
            _load_checkpoint(self.network, checkpoint, torch)
            self.network.to(self.device).requires_grad_(False).eval()
            self.dataset_options = {
                "rgb_root": cfg.rgb_root_folder, "rgb_format": cfg.rgb_format,
                "x_root": cfg.x_root_folder, "x_format": cfg.x_format, "x_single_channel": cfg.x_is_single_channel,
                "gt_root": cfg.gt_root_folder, "gt_format": cfg.gt_format, "transform_gt": cfg.gt_transform,
                "class_names": cfg.class_names, "train_source": cfg.train_source,
            }
            initial = self.dataset("development")[0]
            self.evaluator = Evaluator(initial, 9, cfg.norm_mean, cfg.norm_std, self.network,
                                       [1], False, [self.device.index or 0])
            self.evaluator.val_func = self.network
            self.evaluator._x_inferred_3ch = True
            self.evaluator._use_amp = False

    def dataset(self, split):
        if split not in self.datasets:
            ids = split_ids(split)
            options = dict(self.dataset_options, eval_source=str(split_path(split)))
            # ValPre also reads train images, giving a fixed unaugmented input before course noise.
            dataset = self.dataset_class(options, "val", self.preprocess_class())
            if len(dataset) != len(ids):
                raise ValueError(f"Unexpected loader length for {split}.")
            self.datasets[split] = dataset, {name: pos for pos, name in enumerate(ids)}
        return self.datasets[split]

    def read(self, split, image_id):
        dataset, positions = self.dataset(split)
        item = dataset[positions[image_id]]
        rgb, thermal, label = (np.asarray(item[key]) for key in ("data", "modal_x", "label"))
        image_shape = self.image_size + (3,)
        if item["fn"] != image_id or rgb.shape != image_shape or thermal.shape != image_shape:
            raise ValueError(f"Unexpected image ID or array shape for {image_id}.")
        if label.shape != self.image_size or not np.isin(label, [*range(9), 255]).all():
            raise ValueError(f"Unexpected label shape or class IDs for {image_id}.")
        return rgb, thermal, label

    def tensors(self, rgb, thermal):
        x, t = self.evaluator.process_image_rgbX(rgb, thermal, crop_size=None)
        return tuple(self.torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)).to(self.device)
                     for image in (x, t))

    def forward(self, rgb, thermal):
        with self.torch.no_grad():
            result = self.network(*self.tensors(rgb, thermal))
        self.check_logits(result)
        return result

    def check_logits(self, result):
        if (not isinstance(result, self.torch.Tensor) or tuple(result.shape) != (1, 9, *self.image_size)
                or not self.torch.isfinite(result).all().item()):
            raise ValueError("Expected finite FP32 logits with shape [1, 9, 480, 640].")

    def tapped_forward(self, tap, rgb, thermal):
        features, low, native = tap.forward(*self.tensors(rgb, thermal))
        self.check_logits(native)
        return features, low, native

    def close(self):
        queue = getattr(self.evaluator, "results_queue", None)
        if queue is not None:
            queue.close()


class FeatureTap:
    def __init__(self, runtime):
        self.network = runtime.network
        layer = self.network.decode_head.linear_pred
        if (not isinstance(layer, runtime.torch.nn.Conv2d) or layer.in_channels != 512
                or layer.out_channels != 9 or layer.kernel_size != (1, 1)):
            raise ValueError("Expected the original 512-to-9 decoder classifier.")
        self.captured = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.captured = inputs[0].detach(), output.detach()

    def forward(self, rgb, thermal):
        import torch
        self.captured = None
        self.network.eval()
        with torch.no_grad():
            native = self.network(rgb, thermal)
        if self.captured is None:
            raise RuntimeError("Original decoder hook did not capture features.")
        features, low = self.captured
        self.captured = None
        return features, low, native

    def close(self):
        self.handle.remove()
        self.captured = None


def new_heads():
    """Create zero initialized B (one head) and C (two scene heads)."""
    from torch import nn
    heads = nn.Module()
    heads.unified = nn.Conv2d(512, 9, 1)
    heads.experts = nn.ModuleDict({scene: nn.Conv2d(512, 9, 1) for scene in ("D", "N")})
    for layer in (heads.unified, *heads.experts.values()):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    return heads


def residual_logits(heads, features, low, family, scene, size, gamma):
    from torch.nn.functional import interpolate
    if family not in ("B", "C") or scene not in ("D", "N") or gamma not in GAMMAS:
        raise ValueError("Invalid residual family, scene or strength.")
    layer = heads.unified if family == "B" else heads.experts[scene]
    with_residual = low.detach() + float(gamma) * layer(features.detach())
    return interpolate(with_residual, size=size, mode="bilinear", align_corners=False)


def head_arrays(heads):
    return {key: value.detach().cpu().numpy().copy() for key, value in heads.state_dict().items()}


def load_heads(heads, arrays, device):
    import torch
    own = heads.state_dict()
    if set(own) != set(arrays):
        raise ValueError("Unexpected residual tensor names.")
    for key, value in arrays.items():
        if value.shape != tuple(own[key].shape) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid residual tensor: {key}")
    heads.load_state_dict({k: torch.from_numpy(v.copy()).to(device) for k, v in arrays.items()}, strict=True)
    return heads


def load_bundle(path, device):
    heads = new_heads().to(device)
    with np.load(path, allow_pickle=False) as data:
        tensor_names = {"head__" + key for key in heads.state_dict()}
        required = {"format", "B_epoch", "B_gamma", "C_epoch", "C_gamma"} | tensor_names
        if set(data.files) != required or str(data["format"]) != BUNDLE_FORMAT:
            raise ValueError("Unsupported residual bundle.")
        choices = {m: (int(data[f"{m}_epoch"]), float(data[f"{m}_gamma"])) for m in ("B", "C")}
        for epoch, gamma in choices.values():
            if epoch not in (0, *EPOCHS) or gamma not in GAMMAS or (epoch == 0) != (gamma == 0):
                raise ValueError("Invalid residual epoch or strength.")
        load_heads(heads, {key: data[f"head__{key}"] for key in heads.state_dict()}, device)
    return heads.requires_grad_(False).eval(), choices


def final_logits(runtime, tap, heads, choices, rgb, thermal, label, scene, families=FAMILIES):
    with runtime.torch.no_grad():
        features, low, native = runtime.tapped_forward(tap, rgb, thermal)
        outputs = {}
        for family in families:
            gamma = 0.0 if family == "A" else choices[family][1]
            outputs[family] = native if gamma == 0 else residual_logits(
                heads, features, low, family, scene, label.shape, gamma)
            runtime.check_logits(outputs[family])
    return outputs


def prepare_inputs(rgb, thermal, condition, image_id, seed, restoration):
    rgb, thermal = apply_corruption(rgb, thermal, condition, image_id, seed)
    if restoration != "RAW":
        thermal = restore_thermal(thermal, restoration)[0]
    return rgb, thermal


def logits_numpy(logits):
    return logits[0].detach().permute(1, 2, 0).float().cpu().numpy()
