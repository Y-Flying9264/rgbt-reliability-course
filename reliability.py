"""Pixel metrics and CLEAN-only temperature fitting used in the course experiments."""
import numpy as np

from degradations import stable_seed

TEMPERATURE_BOUNDS = (0.05, 20.0)
CALIBRATION_PIXELS_PER_IMAGE = 4096
# Keep the archived sampling key to reproduce the fixed pixel sample.
CALIBRATION_PURPOSE = "batch05_calibration_pixels"


def weighted_auroc(error, score, weight=None):
    """Weighted Mann-Whitney AUROC: wrong prediction is the positive class."""
    error = np.asarray(error, dtype=bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    weight = np.ones(len(error), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
    if not (len(error) == len(score) == len(weight)):
        raise ValueError("AUROC arrays have different lengths.")
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("AUROC requires finite scores and positive weights.")
    pos_total = weight[error].sum()
    neg_total = weight[~error].sum()
    if pos_total == 0 or neg_total == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    s, y, w = score[order], error[order], weight[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    positives = np.add.reduceat(w * y, starts)
    negatives = np.add.reduceat(w * ~y, starts)
    lower_negatives = np.cumsum(negatives) - negatives
    numerator = np.sum(positives * (lower_negatives + 0.5 * negatives))
    return float(numerator / (pos_total * neg_total))


class CalibrationStats:
    def __init__(self, bins=15):
        self.bins = int(bins)
        self.count = np.zeros(self.bins, dtype=np.int64)
        self.correct = np.zeros(self.bins, dtype=np.int64)
        self.conf_sum = np.zeros(self.bins, dtype=np.float64)
        self.nll_sum = 0.0
        self.n = 0
        self.error_count = 0
        self.hist = {name: np.zeros(30, dtype=np.int64) for name in [
            "msp_correct", "msp_wrong", "entropy_correct", "entropy_wrong"]}

    def update(self, confidence, entropy, correct, nll):
        confidence = np.asarray(confidence)
        correct = np.asarray(correct, dtype=bool)
        ids = np.minimum((np.clip(confidence, 0, 1) * self.bins).astype(np.int64), self.bins - 1)
        self.count += np.bincount(ids, minlength=self.bins)
        self.correct += np.bincount(ids, weights=correct.astype(np.int64), minlength=self.bins).astype(np.int64)
        self.conf_sum += np.bincount(ids, weights=confidence, minlength=self.bins)
        self.n += len(confidence)
        self.error_count += int((~correct).sum())
        self.nll_sum += float(np.sum(nll, dtype=np.float64))
        for kind, values in [("msp", confidence), ("entropy", entropy)]:
            for suffix, mask in [("correct", correct), ("wrong", ~correct)]:
                self.hist[kind + "_" + suffix] += np.histogram(values[mask], bins=30, range=(0, 1))[0]

    def finish(self):
        accuracy = np.divide(self.correct, self.count, out=np.full(self.bins, np.nan), where=self.count > 0)
        confidence = np.divide(self.conf_sum, self.count, out=np.full(self.bins, np.nan), where=self.count > 0)
        ece = float(np.nansum(np.abs(self.correct - self.conf_sum)) / self.n) if self.n else None
        return {"pixels": self.n, "error_pixels": self.error_count,
                "error_rate": self.error_count / self.n if self.n else None,
                "ECE": ece, "NLL": self.nll_sum / self.n if self.n else None,
                "bin_count": self.count, "bin_accuracy": accuracy, "bin_confidence": confidence,
                "score_histograms": self.hist}


class MetricsAccumulator:
    def __init__(self, classes=9, bins=15):
        self.classes = int(classes)
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.groups = {"V": CalibrationStats(bins), "F": CalibrationStats(bins)}
        self.samples = []

    def update(self, probability, label, indices, image_id, log_probability=None):
        p = np.asarray(probability)
        label = np.asarray(label)
        if p.shape != label.shape + (self.classes,):
            raise ValueError(f"Probability/label shapes differ: {p.shape}, {label.shape}")
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1.00001:
            raise ValueError("Invalid final probabilities.")
        if float(np.max(np.abs(p.sum(axis=-1) - 1))) > 1e-4:
            raise ValueError("Final probabilities do not sum to one.")
        if log_probability is None:
            logp = np.log(np.clip(p.astype(np.float64), np.finfo(np.float64).tiny, 1))
        else:
            logp = np.asarray(log_probability)
            if logp.shape != p.shape or not np.isfinite(logp).all():
                raise ValueError("Invalid final log-probabilities.")
        prediction = p.argmax(axis=-1)
        confidence = p.max(axis=-1)
        entropy = np.clip(-np.einsum("...c,...c->...", p, logp, dtype=np.float64) / np.log(self.classes), 0, 1)
        y = label.reshape(-1).astype(np.int64)
        pred = prediction.reshape(-1)
        conf = confidence.reshape(-1)
        ent = entropy.reshape(-1)
        valid = (y >= 0) & (y < self.classes)
        foreground = valid & (y > 0)
        correct = pred == y
        nll = np.zeros(y.shape, dtype=np.float64)
        row = np.flatnonzero(valid)
        nll[row] = -logp.reshape(-1, self.classes)[row, y[row]]
        self.confusion += np.bincount(self.classes * y[valid] + pred[valid], minlength=self.classes ** 2).reshape(self.classes, self.classes)
        for name, mask in [("V", valid), ("F", foreground)]:
            self.groups[name].update(conf[mask], ent[mask], correct[mask], nll[mask])
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(y) or not valid[indices].all()):
            raise ValueError("AUROC sample contains invalid pixels.")
        sampling_weight = float(valid.sum()) / len(indices) if len(indices) else 1.0
        self.samples.append({"image_id": str(image_id), "pixel_index": indices.astype(np.int32),
                             "msp": conf[indices].astype(np.float32), "entropy": ent[indices],
                             "error": (~correct[indices]).astype(np.uint8),
                             "foreground": foreground[indices].astype(np.uint8),
                             "weight": np.full(len(indices), sampling_weight, dtype=np.float64)})
        return {"prediction": prediction.astype(np.uint8), "confidence": confidence, "entropy": entropy,
                "error": (valid & ~correct).reshape(label.shape).astype(np.uint8)}

    def sample_arrays(self):
        fields = ["pixel_index", "msp", "entropy", "error", "foreground", "weight"]
        result = {key: np.concatenate([s[key] for s in self.samples]) if self.samples else np.array([]) for key in fields}
        result["image_ids"] = np.asarray([s["image_id"] for s in self.samples], dtype=str)
        result["image_index"] = np.concatenate([np.full(len(s["error"]), i, dtype=np.int32) for i, s in enumerate(self.samples)]) if self.samples else np.array([], dtype=np.int32)
        return result

    def finish(self):
        q = self.confusion
        union = q.sum(1) + q.sum(0) - np.diag(q)
        iou = np.divide(np.diag(q), union, out=np.full(self.classes, np.nan), where=union > 0)
        def mean(values):
            values = values[np.isfinite(values)]
            return float(values.mean() * 100) if len(values) else None
        result = {"mIoU9_pct": mean(iou), "mIoU8_pct": mean(iou[1:]), "class_IoU_pct": iou * 100,
                  "confusion_matrix": q, "class_GT_pixels": q.sum(1), "class_pred_pixels": q.sum(0), "groups": {}}
        arrays = self.sample_arrays()
        for name in ["V", "F"]:
            mask = np.ones(len(arrays["error"]), dtype=bool) if name == "V" else arrays["foreground"].astype(bool)
            g = self.groups[name].finish()
            g["AUROC_MSP"] = weighted_auroc(arrays["error"][mask], 1.0 - arrays["msp"][mask].astype(np.float64), arrays["weight"][mask])
            g["AUROC_entropy"] = weighted_auroc(arrays["error"][mask], arrays["entropy"][mask], arrays["weight"][mask])
            g["AUROC_sample_pixels"] = int(mask.sum())
            g["AUROC_sample_error_rate"] = float(arrays["error"][mask].mean()) if mask.any() else None
            result["groups"][name] = g
        return result


def sample_indices(label, n, seed, image_id, class_count=9):
    flat = np.asarray(label).reshape(-1)
    valid = np.flatnonzero((flat >= 0) & (flat < class_count))
    if len(valid) <= n:
        return valid
    rng = np.random.default_rng(stable_seed(seed, image_id, "auroc_pixels"))
    return np.sort(rng.choice(valid, size=int(n), replace=False))


def fit_sample_indices(label, seed, image_id, cap=CALIBRATION_PIXELS_PER_IMAGE):
    label = np.asarray(label).reshape(-1)
    valid = np.flatnonzero((label >= 0) & (label < 9))
    if not len(valid):
        raise ValueError('校准图像没有有效像素。')
    if len(valid) <= cap:
        return valid
    rng = np.random.default_rng(stable_seed(seed, image_id, CALIBRATION_PURPOSE))
    return np.sort(rng.choice(valid, int(cap), replace=False))


def probability_at_temperature(logits, temperature):
    """Float64 log-softmax of FP32 network logits; shared by all schemes."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('温度必须是有限正数。')
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 2 or z.shape[-1] != 9 or not np.isfinite(z).all():
        raise ValueError('logits须为有限的末维九类数组。')
    centered = (z - z.max(axis=-1, keepdims=True)) / temperature
    logp = centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))
    probability = np.exp(logp)
    if not np.isfinite(logp).all():
        raise ValueError('温度变换产生非有限log概率。')
    return probability, logp


def objective_value(logits, labels, weights, inverse_temperature):
    """Weighted NLL and its derivative in beta=1/T, using bounded chunks."""
    beta = float(inverse_temperature)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('逆温度必须为有限正数。')
    total_loss, total_gradient, total_weight = 0.0, 0.0, 0.0
    for first in range(0, len(labels), 32768):
        last = first + 32768
        z = np.array(logits[first:last], dtype=np.float64, copy=True)
        y = np.asarray(labels[first:last], dtype=np.int64)
        w = np.asarray(weights[first:last], dtype=np.float64)
        z -= z.max(axis=1, keepdims=True)
        selected = z[np.arange(len(y)), y]
        scaled = beta * z
        exponent = np.exp(scaled)
        denominator = exponent.sum(axis=1)
        losses = np.log(denominator) - beta * selected
        gradient = (exponent * z).sum(axis=1) / denominator - selected
        total_loss += float(np.dot(w, losses))
        total_gradient += float(np.dot(w, gradient))
        total_weight += float(w.sum())
    if total_weight <= 0:
        raise ValueError('校准权重总和须为正。')
    return total_loss / total_weight, total_gradient / total_weight


def fit_temperature(logits, labels, weights, group):
    """Convex scalar inverse-temperature fit. No evaluation data is accepted."""
    from scipy.optimize import brentq
    z = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if z.ndim != 2 or z.shape[1] != 9 or len(z) != len(labels) or len(z) != len(weights):
        raise ValueError('拟合样本形状不一致。')
    if not np.isfinite(z).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('拟合样本或权重无效。')
    if group not in ['V', 'F']:
        raise ValueError('拟合目标仅支持V或F。')
    valid = (labels >= 0) & (labels < 9)
    mask = valid if group == 'V' else valid & (labels > 0)
    if not mask.any():
        raise ValueError(f'校准样本中没有{group}像素。')
    z, y, w = z[mask], labels[mask].astype(np.int64), weights[mask]
    trace = []
    def evaluate(beta):
        loss, gradient = objective_value(z, y, w, beta)
        trace.append({'inverse_temperature': float(beta), 'temperature': float(1 / beta),
                      'NLL': float(loss), 'gradient_beta': float(gradient)})
        return loss, gradient
    base_loss, g1 = evaluate(1.0)
    lower_beta, upper_beta = 1 / TEMPERATURE_BOUNDS[1], 1 / TEMPERATURE_BOUNDS[0]
    _, gl = evaluate(lower_beta)
    _, gu = evaluate(upper_beta)
    if abs(gl) < 1e-13 and abs(gu) < 1e-13:
        beta, status = 1.0, 'flat_objective_T1'
    elif gl >= 0:
        beta, status = lower_beta, 'temperature_upper_bound'
    elif gu <= 0:
        beta, status = upper_beta, 'temperature_lower_bound'
    elif abs(g1) < 1e-12:
        beta, status = 1.0, 'interior_stationary'
    else:
        beta = float(brentq(lambda x: evaluate(x)[1], lower_beta, upper_beta,
                            xtol=1e-11, rtol=1e-11, maxiter=100))
        status = 'interior_stationary'
    fitted_loss, fitted_gradient = evaluate(beta)
    if fitted_loss > base_loss + 1e-9:
        raise ValueError('拟合后的校准目标NLL高于T=1，请回传拟合记录。')
    return {'fit_group': group, 'temperature': float(1 / beta), 'inverse_temperature': beta,
            'temperature_bounds': list(TEMPERATURE_BOUNDS), 'solver_status': status,
            'boundary_hit': status.startswith('temperature_'), 'sample_pixels': int(mask.sum()),
            'weighted_pixels': float(w.sum()), 'class_sample_counts': np.bincount(y, minlength=9),
            'NLL_before': base_loss, 'NLL_after': fitted_loss, 'gradient_beta_after': fitted_gradient,
            'solver': 'weighted NLL, convex inverse-temperature, derivative root/brentq',
            'trace': trace}


def summarize_extra(result):
    """Read means and high-confidence-error fractions from full pixel bins."""
    q = np.asarray(result['confusion_matrix'])
    for group in ['V', 'F']:
        g = result['groups'][group]
        n = np.asarray(g['bin_count'])
        conf = np.asarray(g['bin_confidence'], dtype=float)
        g['pixel_accuracy'] = 1 - g['error_rate'] if g['pixels'] else None
        g['mean_MSP'] = float(np.nansum(n * conf) / n.sum()) if n.sum() else None
        high = int(sum(g['score_histograms']['msp_wrong'][27:]))
        g['high_confidence_error_pixels'] = high
        g['high_confidence_error_fraction'] = high / g['error_pixels'] if g['error_pixels'] else None
    f_count = int(q[1:].sum())
    result['target_predicted_0_pixels'] = int(q[1:, 0].sum())
    result['target_predicted_0_fraction'] = result['target_predicted_0_pixels'] / f_count if f_count else None
    return result


def metric_rows(split, condition, scheme, temperature, result, images):
    row = {'split': split, 'condition': condition, 'scheme': scheme, 'temperature': temperature,
           'images': images, 'mIoU9_pct': result['mIoU9_pct'], 'mIoU8_pct': result['mIoU8_pct']}
    for group in ['V', 'F']:
        for key in ['pixels', 'error_pixels', 'pixel_accuracy', 'mean_MSP', 'error_rate', 'ECE', 'NLL',
                    'AUROC_MSP', 'AUROC_entropy', 'AUROC_sample_pixels', 'high_confidence_error_fraction']:
            row[f'{key}_{group}'] = result['groups'][group][key]
    row['target_predicted_0_pixels'] = result['target_predicted_0_pixels']
    row['target_predicted_0_fraction'] = result['target_predicted_0_fraction']
    return row


def segmentation(q):
    q = np.asarray(q,dtype=np.int64)
    union = q.sum(0)+q.sum(1)-np.diag(q)
    iou = np.divide(np.diag(q)*100.,union,out=np.full(9,np.nan),where=union>0)
    f,b = int(q[1:].sum()),int(q[0].sum())
    fp = int(q[0,1:].sum())
    pred_f = int(q[:,1:].sum())
    return {'mIoU9_pct':float(np.nanmean(iou)), 'mIoU8_pct':float(np.nanmean(iou[1:])),
        'class_IoU_pct':iou, 'foreground_pixels':f, 'background_pixels':b,
        'target_predicted_0_pixels':int(q[1:,0].sum()),
        'target_predicted_0_fraction':float(q[1:,0].sum()/f) if f else None,
        'background_false_positive_pixels':fp,
        'background_false_positive_fraction':fp/b if b else None,
        'foreground_binary_precision':float(q[1:,1:].sum()/pred_f) if pred_f else None,
        'foreground_class_accuracy':float(np.diag(q)[1:].sum()/f) if f else None}


def paired_transitions(label, raw, candidate):
    y,r,c = np.asarray(label),np.asarray(raw),np.asarray(candidate)
    if y.shape != r.shape or y.shape != c.shape:
        raise ValueError('Paired prediction/GT shapes differ.')
    valid = (y>=0)&(y<9);fg = valid&(y>0);bg = valid&(y==0)
    before,after = r==y,c==y
    result = {'prediction_changed_pixels':int((valid&(r!=c)).sum())}
    for name,mask in [('V',valid),('F',fg),('B',bg)]:
        result[name+'_wrong_to_correct'] = int((mask&~before&after).sum())
        result[name+'_correct_to_wrong'] = int((mask&before&~after).sum())
        result[name+'_both_wrong'] = int((mask&~before&~after).sum())
        result[name+'_both_correct'] = int((mask&before&after).sum())
        result[name+'_net_correct_gain'] = result[name+'_wrong_to_correct']-result[name+'_correct_to_wrong']
    result['target_raw0_to_correct'] = int((fg&(r==0)&after).sum())
    result['target_raw0_to_other_wrong'] = int((fg&(r==0)&(c>0)&~after).sum())
    result['target_new_predicted_0'] = int((fg&(r>0)&(c==0)).sum())
    result['background_new_false_positive'] = int((bg&(r==0)&(c>0)).sum())
    result['background_fixed_false_positive'] = int((bg&(r>0)&(c==0)).sum())
    return result
