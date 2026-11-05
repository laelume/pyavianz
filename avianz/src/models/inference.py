# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean
#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# PyTorch inference utilities


from avianz.src.core import spectrogram
from avianz.src.core import config_loader
from avianz.src.core import audio_data
from avianz.src.utils import shapes
# CIRCULAR from avianz.src.models import inference 

import numpy as np
import os
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from avianz.src.models import gpu_config

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# C O N F I G
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

VERBOSE = False    # toggle verbose debug logging on/off

logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)


def generate_features_threaded(segment_args, feature_fn, max_workers=None):
    """ Generate NN input features for multiple segments concurrently via a thread pool.

    segment_args: ordered list of argument-tuples, one per segment, to be unpacked into feature_fn
    feature_fn: callable performing the CPU-bound feature generation for a single segment
                (e.g. PostProcess.generate_nn_features)
    max_workers: worker count; falls back to gpu_config.CPU_THREAD_COUNT, then
                 gpu_config.available_cpu_count() (CPUs allocated to this process, not the
                 whole machine), keeping this pool's size consistent with the
                 thread count set for torch's own intra-op parallelism

    Order of results matches segment_args (uses pool.map rather than as-completed), which
    keeps downstream indexing against the caller's segment list aligned.
    """
    workers = max_workers or gpu_config.CPU_THREAD_COUNT or gpu_config.available_cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda args: feature_fn(*args), segment_args))
    return results


def predict_batch(model, features):
    """ Run batch prediction using the model. """
    model.eval()
    with torch.no_grad():
        if isinstance(features, torch.Tensor):
            tensor_input = features
        else:
            tensor_input = torch.from_numpy(features).float()

        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if len(tensor_input.shape) == 4 and tensor_input.shape[-1] == 1:
            tensor_input = tensor_input.permute(0, 3, 1, 2)
        elif len(tensor_input.shape) == 3:
            tensor_input = tensor_input.unsqueeze(1)

        device = next(model.parameters()).device
        tensor_input = tensor_input.to(device)

        predictions = model(tensor_input)
        return predictions.cpu().numpy()



