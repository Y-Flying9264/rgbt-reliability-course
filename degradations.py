"""Course RGB/T perturbations and thermal restoration in display units [0, 255]."""
import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

METHODS = ("RAW", "GAUSS_0P8", "ADAPT_WIENER_5")
CONDITIONS = {"CLEAN": {"id": "CLEAN", "kind": "clean", "value": None, "repeat": "r0"}}
for prefix, kind, levels in (("RGB-DARK", "rgb_dark", (.7, .4, .2)),
                             ("RGB-BLUR", "rgb_blur", (1., 2., 4.)),
                             ("T-NOISE", "t_noise", (.02, .05, .10))):
    for level, value in enumerate(levels, 1):
        cid = f"{prefix}-L{level}"
        CONDITIONS[cid] = {"id": cid, "kind": kind, "value": value, "repeat": "r0"}
for level in (2, 3):
    for repeat in ("r1", "r2"):
        base = CONDITIONS[f"T-NOISE-L{level}"]
        cid = f"{base['id']}-{repeat}"
        CONDITIONS[cid] = dict(base, id=cid, repeat=repeat)
FINAL_CONDITIONS = ("CLEAN", "T-NOISE-L2", "T-NOISE-L3")


def stable_seed(seed, image_id, purpose):
    raw = f"{int(seed)}|{image_id}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def apply_corruption(rgb, thermal, condition, image_id, seed):
    """Input/output: HWC images in display units [0,255], before normalization."""
    rgb_out = np.asarray(rgb, dtype=np.float32).copy()
    thermal_out = np.asarray(thermal, dtype=np.float32).copy()
    kind = condition["kind"]
    if kind == "clean":
        return rgb_out, thermal_out
    if kind == "rgb_dark":
        rgb_out *= float(condition["value"])
    elif kind == "rgb_blur":
        from scipy.ndimage import gaussian_filter
        sigma = float(condition["value"])
        rgb_out = gaussian_filter(rgb_out, sigma=(sigma, sigma, 0), mode="reflect", truncate=3.0)
    elif kind == "t_noise":
        if thermal_out.ndim == 3:
            if thermal_out.shape[2] != 3 or not (
                np.array_equal(thermal_out[..., 0], thermal_out[..., 1])
                and np.array_equal(thermal_out[..., 0], thermal_out[..., 2])
            ):
                raise ValueError("MFNet thermal input must contain repeated grayscale channels.")
            gray = thermal_out[..., 0]
        else:
            gray = thermal_out
        repeat = condition.get("repeat", "r0")
        rng = np.random.default_rng(stable_seed(seed, image_id, "thermal_noise_" + repeat))
        noise = rng.standard_normal(gray.shape)
        gray = np.clip(gray + 255.0 * float(condition["value"]) * noise, 0, 255).astype(np.float32)
        thermal_out = np.repeat(gray[..., None], 3, axis=2) if thermal_out.ndim == 3 else gray
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return np.clip(rgb_out, 0, 255), thermal_out


def grayscale(thermal):
    a = np.asarray(thermal)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('Thermal array must have three repeated grayscale channels.')
    if not np.array_equal(a[...,0], a[...,1]) or not np.array_equal(a[...,0], a[...,2]):
        raise ValueError('Thermal channels are not repeated grayscale.')
    if min(a.shape[:2]) < 6 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError('Invalid thermal display intensities.')
    return np.array(a[...,0], dtype=np.float64, copy=True)


def estimate_noise_sigma(gray):
    """MAD of unit-energy disjoint 2x2 diagonal Haar coefficients, display units."""
    a = np.asarray(gray,dtype=np.float64)
    h,w = (a.shape[0]//2)*2, (a.shape[1]//2)*2
    a = a[:h,:w]
    hh = (a[0::2,0::2]-a[0::2,1::2]-a[1::2,0::2]+a[1::2,1::2])/2
    return float(np.median(np.abs(hh-np.median(hh)))/0.6744897501960817)


def restore_thermal(thermal, method):
    """Filters receive only the observed thermal image and a fixed method ID."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    gray = grayscale(thermal)
    sigma = estimate_noise_sigma(gray)
    if method == 'RAW':
        out = gray
    elif method == 'GAUSS_0P8':
        out = gaussian_filter(gray,sigma=0.8,mode='reflect',truncate=3.0)
    else:
        mean = uniform_filter(gray,size=5,mode='reflect')
        variance = np.maximum(uniform_filter(gray*gray,size=5,mode='reflect')-mean*mean,0.)
        gain = np.maximum(variance-sigma*sigma,0.)/np.maximum(variance,1e-12)
        out = mean + gain*(gray-mean)
    out = np.clip(out,0,255).astype(np.float32)
    restored = np.repeat(out[...,None],3,axis=-1)
    return restored, {
        'estimated_sigma_display':sigma,
        'estimated_sigma_0to1':sigma/255.,
        'mean_absolute_input_change':float(np.mean(np.abs(out.astype(np.float64)-gray))),
        'input_change_RMS':float(np.sqrt(np.mean((out.astype(np.float64)-gray)**2))),
        'observed_thermal_mean':float(gray.mean()),
        'restored_thermal_mean':float(out.astype(np.float64).mean()),
        'method':method,
    }
