import sys
import random
import logging
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np
import torch

from dexonomy.op import *

seed = 12
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    try:
        eval(f"operate_{cfg.op_name}")(cfg)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
        sys.exit(1)
    return


if __name__ == "__main__":
    main()
