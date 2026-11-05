# src/models/gpu_config.py
# Version 0.1 07/07/26
# Author: laelume

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2026

# GPU/CPU device selection and environment diagnostics for NN inference

import os
import logging
import torch
import sys

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# C O N F I G
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

VERBOSE = False            # toggle verbose debug logging on/off
CPU_THREAD_COUNT = None    # None, falls back to available_cpu_count() at runtime (allocated CPUs, not all cores)

INSTALL_MODE = os.environ.get("AVIANZ_INSTALL_MODE", "auto")
# INSTALL_MODE sets which install-time expectation is checked at runtime:
#   "gpu"  : expects a CUDA build of torch and a reachable GPU, raises a clear
#            error at startup if either is missing instead of falling through
#            to the cpu prompt
#   "cpu"  : skips GPU checks entirely, same effect as passing useGpu=False
#   "auto" : original behaviour, attempts GPU if available, prompts for cpu
#            fallback if not
# Set via environment variable before launch, e.g.:
#   $env:AVIANZ_INSTALL_MODE = "gpu"


logger = logging.getLogger("gpu_config")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)


def check_cuda_environment():
    """ Check whether torch has CUDA support built in, and whether a GPU is reachable at runtime.

    torch.version.cuda is None means the installed torch build has no CUDA support,
    regardless of hardware. torch.cuda.is_available() means a GPU is reachable given
    that build. These are separate checks: a CPU-only build always fails
    is_available() with no indication why, while a CUDA build failing is_available()
    points to a driver or hardware problem instead of a packaging problem.

    Returns a tuple: (torchHasCudaBuild, gpuReachable)
    """
    torchHasCudaBuild = torch.version.cuda is not None
    gpuReachable = torch.cuda.is_available()

    if not torchHasCudaBuild:
        logger.debug("torch build has no CUDA support (torch.version.cuda is None)")
    elif not gpuReachable:
        logger.debug("torch build has CUDA support, but no GPU is reachable")
    else:
        logger.debug("torch build has CUDA support and a GPU is reachable")

    return torchHasCudaBuild, gpuReachable


def print_environment_diagnostics():
    """ Print Python executable path and torch/CUDA build info at startup, so environment mismatches are visible without needing manual checks each time.

    Covers the two most common causes of a CUDA build appearing missing when it
    was actually installed correctly elsewhere: running from a different Python
    environment than the one the CUDA torch build was installed into, and torch
    being shadowed by a separate conda-installed build. Runs automatically as
    part of configure_gpu_memory rather than requiring separate manual commands.
    """
    print(f"Python executable: {sys.executable}")
    print(f"torch location: {torch.__file__}")
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    logger.debug("Environment diagnostics printed: executable=%s, torch=%s, cuda_build=%s",
                 sys.executable, torch.__file__, torch.version.cuda)

def configure_gpu_memory(model=None, verbose=None, useGpu=True):
    """ Select and configure inference device (GPU if available, else CPU), and move model onto it.

    Preserves cache-clearing behaviour when called with no model (existing call
    sites keep working unchanged), and returns the resolved torch.device so
    callers can track where inference is executing. Moves model onto device.

    useGpu=False skips the cuda check and y/n prompt, forcing cpu directly. This is
    an explicit setting, not an unexpected fallback, so no prompt is shown.

    When useGpu is True but no GPU is detected, behaviour depends on INSTALL_MODE:
      "gpu"  : raises immediately, states whether the issue is a missing CUDA build
               or an unreachable GPU, since gpu mode was explicitly requested and a
               silent or prompted fallback would hide a setup problem
      "cpu"  : forces cpu directly, same as useGpu=False
      "auto" : prompts at command line for y/n confirmation before proceeding on cpu
    """
    if verbose is not None:
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    if not useGpu or INSTALL_MODE == "cpu":
        # FALLBACK: explicit cpu forcing, engaged when useGpu=False or
        # INSTALL_MODE="cpu" is set. Skips cuda check and y/n prompt.
        device = torch.device("cpu")
        print("GPU usage disabled by request, proceeding with CPU")
        logger.debug("useGpu=False or INSTALL_MODE=cpu, forcing cpu without prompt")
        configure_cpu_threads()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.debug("GPU detected, running inference on cuda device")
    elif INSTALL_MODE == "gpu":
        # GPU mode was explicitly requested at install time, so a missing GPU here
        # is a setup problem, not an expected fallback. Raises with a specific
        # cause instead of prompting or continuing silently on cpu.
        
        print_environment_diagnostics()
        torchHasCudaBuild, _ = check_cuda_environment()
        if not torchHasCudaBuild:
            raise RuntimeError(
                "INSTALL_MODE=gpu but torch has no CUDA support built in "
                "(torch.version.cuda is None). Reinstall torch with a CUDA build "
                "matching the driver's supported CUDA version (check with "
                "nvidia-smi), e.g. pip install torch --index-url "
                "https://download.pytorch.org/whl/cu126"
            )
        else:
            raise RuntimeError(
                "INSTALL_MODE=gpu and torch has CUDA support, but no GPU is "
                "reachable at runtime. Check nvidia-smi detects the device, and "
                "confirm this environment matches where CUDA torch was installed."
            )
    else:
        # FALLBACK: cpu execution path, engaged when cuda is unavailable and
        # INSTALL_MODE is "auto". Requires y/n confirmation before proceeding,
        # rather than silent fallback.

        print_environment_diagnostics()
        torchHasCudaBuild, _ = check_cuda_environment()

        if not torchHasCudaBuild:
            print("No CUDA-capable torch build detected (torch.version.cuda is "
                  "None). This is a packaging issue, not a hardware issue.")
        else:
            print("torch has CUDA support, but no GPU is reachable at runtime. "
                  "Check nvidia-smi and confirm the active environment matches "
                  "where CUDA torch was installed.")

        response = input("No GPU detected. Proceed with CPU? [y/n]: ").strip().lower()

        while response not in ("y", "n"):
            response = input("Please enter 'y' or 'n': ").strip().lower()

        if response == "n":
            logger.debug("User declined cpu fallback, aborting")
            raise RuntimeError("No GPU detected and CPU fallback declined")

        device = torch.device("cpu")
        print("Proceeding with CPU")
        logger.debug("Proceeding with cpu device after user confirmation")
        configure_cpu_threads()

    if model is not None:
        model.to(device)
        return model, device

    return device


def available_cpu_count():
    """ Number of CPUs this process may actually use.

    os.cpu_count() is every core on the machine (256 on a Rāpoi AMD node), not what a Slurm
    job was allocated; sizing thread pools from it makes torch start hundreds of threads,
    whose reserved memory exceeds the job's --mem (which on Rāpoi also caps virtual memory)
    and fails with "DefaultCPUAllocator: can't allocate memory".
    Order: OMP_NUM_THREADS if set, then the CPUs in this process's affinity mask
    (what Slurm/taskset allow), then os.cpu_count().
    """
    env = os.environ.get("OMP_NUM_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        n = os.process_cpu_count()
        if n:
            return n
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def configure_cpu_threads(num_threads=None):
    """ Set the number of CPU threads torch uses for intra-op parallelism.

    Applies on the cpu-only fallback path, and also alongside GPU inference for any
    CPU-bound pre/post-processing torch ops that run outside the model forward pass.
    Falls back to available_cpu_count() if num_threads is not given and CPU_THREAD_COUNT is unset.
    """
    threads = num_threads or CPU_THREAD_COUNT or available_cpu_count()
    torch.set_num_threads(threads)
    logger.debug("torch cpu thread count set to %d", threads)




# U S A G I
#
# from avianz.src.models import gpu_config
#
# model, device = gpu_config.configure_gpu_memory(model, verbose=True, useGpu=True)
# print(f"inference running on: {device}")
#
# $env:AVIANZ_INSTALL_MODE = "gpu"   # set before launch, enforce GPU-only, raises on setup problems
# $env:AVIANZ_INSTALL_MODE = "cpu"   # forces cpu regardless of hardware, skips all GPU checks
# (unset, or "auto")                 # original behaviour: attempts GPU, prompts for cpu fallback