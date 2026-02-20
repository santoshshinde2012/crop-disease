"""
Model profiling — latency, throughput, and size metrics.

Measures CPU and GPU inference speed and model disk size
for deployment comparison.
"""

import logging
import os
import tempfile
import time
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = None,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Union[int, float]]:
    """Profile model performance metrics.

    Measures CPU latency, GPU latency (if available), and model size.
    The model is restored to its original device after profiling.

    Args:
        model: The model to profile.
        input_size: Input tensor shape (batch, channels, height, width).
        device: Device for profiling.
        num_warmup: Warmup iterations (discarded).
        num_runs: Timed iterations.

    Returns:
        Dict with keys: total_params, model_size_mb,
        cpu_latency_mean_ms, cpu_latency_p95_ms,
        gpu_latency_mean_ms (if CUDA), gpu_latency_p95_ms (if CUDA).
    """
    if device is None:
        device = torch.device("cpu")

    # Remember original device so we can restore at the end
    original_device = next(model.parameters()).device

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Model size on disk
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(model.state_dict(), tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.unlink(tmp_path)

    result = {
        "total_params": total_params,
        "model_size_mb": round(model_size_mb, 2),
    }

    # CPU Latency
    model.cpu().eval()
    dummy_input_cpu = torch.randn(*input_size)

    # Warmup
    for _ in range(num_warmup):
        model(dummy_input_cpu)

    # Timed runs
    cpu_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model(dummy_input_cpu)
        end = time.perf_counter()
        cpu_times.append((end - start) * 1000)  # ms

    result["cpu_latency_mean_ms"] = round(float(np.mean(cpu_times)), 2)
    result["cpu_latency_p95_ms"] = round(float(np.percentile(cpu_times, 95)), 2)

    logger.info(
        "CPU latency: mean=%.2f ms, p95=%.2f ms",
        result["cpu_latency_mean_ms"],
        result["cpu_latency_p95_ms"],
    )

    # GPU Latency (only if CUDA available)
    if torch.cuda.is_available():
        model.to("cuda").eval()
        dummy_input_gpu = torch.randn(*input_size, device="cuda")

        # Warmup
        for _ in range(num_warmup):
            model(dummy_input_gpu)
        torch.cuda.synchronize()

        # Timed runs with CUDA events
        gpu_times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(dummy_input_gpu)
            end_event.record()
            torch.cuda.synchronize()

            gpu_times.append(start_event.elapsed_time(end_event))

        result["gpu_latency_mean_ms"] = round(float(np.mean(gpu_times)), 2)
        result["gpu_latency_p95_ms"] = round(float(np.percentile(gpu_times, 95)), 2)

        logger.info(
            "GPU latency: mean=%.2f ms, p95=%.2f ms",
            result["gpu_latency_mean_ms"],
            result["gpu_latency_p95_ms"],
        )

    # Restore model to its original device
    model.to(original_device)

    return result
