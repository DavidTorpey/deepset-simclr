import logging
import os

import torch

from src.cfg import Config
from src.constants import LATEST_MODEL_FILE_NAME


def restore_from_checkpoint(
        config: Config, model, optimiser
) -> dict:
    to_restore = {'epoch': 0}

    restart_from_checkpoint(
        os.path.join(config.general.output_dir, LATEST_MODEL_FILE_NAME),
        run_variables=to_restore,
        optimiser=optimiser,
        model=model,
    )

    return to_restore


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    logging.info("Found checkpoint at %s", ckp_path)

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logging.info("=> loaded '%s' from checkpoint '%s' with msg %s", key, ckp_path, msg)
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logging.info("=> loaded '%s' from checkpoint: '%s'", key, ckp_path)
                except ValueError:
                    print("=> failed to load '%s' from checkpoint: '%s'", key, ckp_path)
        else:
            print("=> key '%s' not found in checkpoint: '%s'", key, ckp_path)

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
