from __future__ import annotations

import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def build_profiler(cfg: Any):
    """Return a torch profiler for cfg, or None when profiling is disabled.

    cfg must expose: profiler_enabled, profiler_output_dir, profiler_profile_memory.

    Uses a continuous schedule (wait=0, warmup=1, active=50, repeat=0) that
    flushes a gzip-compressed TensorBoard trace to disk every 50 inference steps,
    keeping in-memory trace data bounded while covering the entire run.

    View results:
        tensorboard --logdir <profiler_output_dir>
    """
    if not getattr(cfg, 'profiler_enabled', False):
        return None

    try:
        import torch
        from torch.profiler import (
            ProfilerActivity,
            profile,
            schedule,
            tensorboard_trace_handler,
        )
    except ImportError:
        logger.warning('torch not available — profiler disabled')
        return None

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    output_dir = pathlib.Path(
        getattr(cfg, 'profiler_output_dir', './experiments/profiler')
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    return profile(
        activities=activities,
        schedule=schedule(skip_first=0, wait=0, warmup=1, active=50, repeat=0),
        on_trace_ready=tensorboard_trace_handler(str(output_dir), use_gzip=True),
        record_shapes=False,
        profile_memory=getattr(cfg, 'profiler_profile_memory', False),
        with_stack=False,
    )
