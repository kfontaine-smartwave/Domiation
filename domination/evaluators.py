"""Position evaluators: features [B, F] -> log-odds that "me" wins the game.

Evaluators return log-odds (logits) rather than probabilities: when one side is far ahead every
option is "about 99 %", and only log-odds keep the options apart (the NPU computes in 16-bit
floats, which cannot resolve differences that close to 1.0).

* `LinearEvaluator`   hand-written formula, used to bootstrap self-play (generation 0).
* `MLPEvaluator`      the trained network run with numpy (training workers, tests).
* `OpenVINOEvaluator` the same network compiled by OpenVINO for the NPU (live agent).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_WEIGHTS = MODEL_DIR / "value.npz"
DEFAULT_IR = MODEL_DIR / "value.xml"
NPU_BATCH = 256          # the NPU needs a static input shape: batches are padded to this size

_HAND_WEIGHTS = {
    "army_logratio": 5.0, "income_logratio": 5.0,
    "terr_me": 6.0, "terr_rival": -6.0,
    "region_potential_me": 2.0, "region_potential_rival": -2.0,
    "bonus_at_risk_me": -3.0, "bonus_at_risk_rival": 2.0,
    "threat_cells": -6.0, "opportunity_enemy": 3.0, "opportunity_neutral": 1.5,
    "idle_army_frac": -1.0, "weak_border_frac": -1.5, "revolt_exposure": -2.0,
    "endgame_lead": 8.0, "pending_bonus": 1.0,
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -30, 30)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


class LinearEvaluator:
    name = "hand-written formula (CPU)"

    def __init__(self, weights: dict | None = None, noise: float = 0.0, rng=None):
        w = dict(_HAND_WEIGHTS if weights is None else weights)
        self.w = np.array([w.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
        if noise and rng is not None:      # perturbed copies diversify self-play opponents
            self.w = self.w * (1.0 + noise * rng.standard_normal(N_FEATURES).astype(np.float32))

    def logits(self, feats: np.ndarray) -> np.ndarray:
        return feats @ self.w


class MLPEvaluator:
    name = "value network (numpy, CPU)"

    def __init__(self, weights=DEFAULT_WEIGHTS):
        data = np.load(weights) if not isinstance(weights, dict) else weights
        self.mean = data["mean"].astype(np.float32)
        self.inv_std = (1.0 / data["std"]).astype(np.float32)
        n_layers = sum(1 for k in data.keys() if k.startswith("W"))
        self.layers = [(data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32))
                       for i in range(n_layers)]

    def logits(self, feats: np.ndarray) -> np.ndarray:
        h = (feats - self.mean) * self.inv_std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                np.maximum(h, 0.0, out=h)
        return h[:, 0]


def export_openvino(weights=DEFAULT_WEIGHTS, xml_path=DEFAULT_IR, batch: int = NPU_BATCH) -> Path:
    """Write the network as an OpenVINO IR with a static [batch, N_FEATURES] input.
    The graph is built directly from the numpy weights (no PyTorch / ONNX needed)."""
    import openvino as ov
    import openvino.opset13 as ops

    data = np.load(weights)
    n_layers = sum(1 for k in data.keys() if k.startswith("W"))
    x = ops.parameter([batch, N_FEATURES], np.float32, name="features")
    h = ops.multiply(ops.subtract(x, ops.constant(data["mean"].astype(np.float32).reshape(1, -1))),
                     ops.constant((1.0 / data["std"]).astype(np.float32).reshape(1, -1)))
    for i in range(n_layers):
        h = ops.add(ops.matmul(h, ops.constant(data[f"W{i}"].astype(np.float32)), False, False),
                    ops.constant(data[f"b{i}"].astype(np.float32).reshape(1, -1)))
        if i < n_layers - 1:
            h = ops.relu(h)
    model = ov.Model([ops.result(h, name="win_logit")], [x], "domination_value_net")
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(xml_path), compress_to_fp16=False)
    return xml_path


class OpenVINOEvaluator:
    """Runs the value network through OpenVINO, on the NPU when there is one."""

    def __init__(self, xml_path=DEFAULT_IR, device: str = "NPU", fallback: bool = True):
        import openvino as ov

        core = ov.Core()
        available = core.available_devices
        order = [device] + ([d for d in ("NPU", "GPU", "CPU") if d != device] if fallback else [])
        self.device = None
        errors = []
        for dev in order:
            if not any(a == dev or a.startswith(dev + ".") for a in available):
                errors.append(f"{dev}: not present")
                continue
            try:
                model = core.read_model(str(xml_path))
                cache = Path(xml_path).parent / "cache"
                cache.mkdir(exist_ok=True)
                t0 = time.perf_counter()
                self.compiled = core.compile_model(model, dev, {"CACHE_DIR": str(cache)})
                self.compile_s = time.perf_counter() - t0
                self.device = dev
                self.device_name = core.get_property(dev, "FULL_DEVICE_NAME")
                break
            except Exception as exc:        # driver missing, unsupported op...: try the next device
                errors.append(f"{dev}: {str(exc).splitlines()[0][:120]}")
        if self.device is None:
            raise RuntimeError("no OpenVINO device could load the model: " + "; ".join(errors))
        self.fallback_reason = "; ".join(errors) if self.device != device else ""
        self.batch = int(self.compiled.input(0).shape[0])
        self.request = self.compiled.create_infer_request()
        self._buf = np.zeros((self.batch, N_FEATURES), dtype=np.float32)
        self.name = f"value network on {self.device} ({self.device_name}) via OpenVINO"
        self.calls = 0
        self.positions = 0
        self.seconds = 0.0

    def logits(self, feats: np.ndarray) -> np.ndarray:
        n = len(feats)
        out = np.empty(n, dtype=np.float32)
        t0 = time.perf_counter()
        for start in range(0, n, self.batch):
            chunk = feats[start:start + self.batch]
            self._buf[:len(chunk)] = chunk
            self._buf[len(chunk):] = 0.0
            res = self.request.infer({0: self._buf})
            out[start:start + len(chunk)] = next(iter(res.values()))[:len(chunk), 0]
            self.calls += 1
        self.positions += n
        self.seconds += time.perf_counter() - t0
        return out


def load_best_evaluator(device: str = "NPU", log=print):
    """NPU network if possible, else the numpy network, else the hand-written formula."""
    if DEFAULT_IR.exists():
        try:
            ev = OpenVINOEvaluator(DEFAULT_IR, device)
            if ev.fallback_reason:
                log(f"! {device} unavailable ({ev.fallback_reason}); using {ev.device} instead")
            return ev
        except Exception as exc:
            log(f"! OpenVINO unavailable ({exc}); falling back to numpy on the CPU")
    if DEFAULT_WEIGHTS.exists():
        return MLPEvaluator(DEFAULT_WEIGHTS)
    log("! no trained model found (run `python -m domination.train`); using the hand-written formula")
    return LinearEvaluator()
