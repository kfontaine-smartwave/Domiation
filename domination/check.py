"""Checks that the value network really runs on the NPU.

    python -m domination.check

Lists the OpenVINO devices, compiles the network on each of them, compares the results with the
plain numpy implementation and times a batch.
"""
from __future__ import annotations

import numpy as np

from .evaluators import DEFAULT_IR, DEFAULT_WEIGHTS, MLPEvaluator, OpenVINOEvaluator
from .features import N_FEATURES


def main():
    import openvino as ov

    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    for dev in core.available_devices:
        print(f"  {dev:4s} {core.get_property(dev, 'FULL_DEVICE_NAME')}")
    if not DEFAULT_IR.exists():
        raise SystemExit(f"{DEFAULT_IR} is missing: run `python -m domination.train` first")
    reference = MLPEvaluator(DEFAULT_WEIGHTS)
    x = reference.mean + reference.mean.std() * np.random.default_rng(0).standard_normal((1024, N_FEATURES)).astype(np.float32)
    expected = reference.logits(x)
    for dev in ("NPU", "GPU", "CPU"):
        try:
            ev = OpenVINOEvaluator(DEFAULT_IR, dev, fallback=False)
        except RuntimeError as exc:
            print(f"{dev}: unavailable ({exc})")
            continue
        ev.logits(x)                                   # warm-up
        ev.seconds, ev.calls = 0.0, 0
        for _ in range(25):
            out = ev.logits(x)
        print(f"{dev}: {ev.device_name} | compiled in {ev.compile_s:.2f} s | "
              f"{ev.seconds / ev.calls * 1000:.2f} ms per batch of {ev.batch} positions | "
              f"largest difference with numpy (log-odds): {np.abs(out - expected).max():.4f}")
    npu = [d for d in core.available_devices if d.startswith("NPU")]
    print("NPU ready: the agent will use it." if npu else "No NPU found: the agent will fall back to GPU or CPU.")


if __name__ == "__main__":
    main()
