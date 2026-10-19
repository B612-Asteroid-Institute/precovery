"""Application-owned Ray initialization for precovery schedulers.

Ray orchestration is an application concern. Keep the behavior that precovery
previously consumed from adam-core while allowing adam-core itself to remain
scheduler-independent.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import ray

logger = logging.getLogger(__name__)

_JAX_FORK_RUNTIMEWARNING_RE = r"os\.fork\(\) was called\..*JAX is multithreaded.*"


def _silence_jax_fork_runtimewarning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_JAX_FORK_RUNTIMEWARNING_RE,
        category=RuntimeWarning,
    )


def initialize_use_ray(
    num_cpus: int | None = None,
    object_store_bytes: int | None = None,
    **kwargs: Any,
) -> bool:
    """Connect to an existing Ray cluster or start one with bounded resources.

    Explicitly serial callers (``num_cpus <= 1``) do not initialize Ray and
    receive ``False``. Parallel or automatic callers first try ``address=auto``
    and then create a local cluster while preserving caller runtime settings.
    """

    if num_cpus is not None and int(num_cpus) <= 1:
        return False

    _silence_jax_fork_runtimewarning()
    kwargs.setdefault("include_dashboard", False)

    if not ray.is_initialized():
        logger.info("Ray is not initialized. Initializing...")
        try:
            logger.info("Attempting to connect to existing Ray cluster...")
            ray.init(address="auto", **kwargs)
        except ConnectionError:
            logger.info("Could not connect to existing Ray cluster.")
            logger.info(
                "Attempting Ray with %s CPUs and %s object-store bytes.",
                num_cpus,
                object_store_bytes,
            )
            ray.init(
                num_cpus=num_cpus,
                object_store_memory=object_store_bytes,
                **kwargs,
            )

    logger.info("Ray resources: %s", ray.cluster_resources())
    return True
