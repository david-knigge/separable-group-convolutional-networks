import random
import os
import torch
import numpy as np


def fix_seed(seed=None):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_group_elem(model, num_elements):
    """
    Modify discrete model's number of group elements (this needs to be easier)
    """
    model.group.__init__(num_elements=num_elements)

    model.lifting.register_buffer("transformed_grids", model.lifting.transform_kernel_grid_by_group(
        model.lifting.grid_Rn
    ))

    model.gconv_spatial_pooling.register_buffer("grid_H", model.gconv_spatial_pooling.get_kernel_grid_H())
    model.gconv_spatial_pooling.register_buffer("transformed_grids", model.gconv_spatial_pooling.transform_kernel_grid_by_group(
        model.gconv_spatial_pooling.grid_Rn, model.gconv_spatial_pooling.grid_H
    ))

    for gconv_block in model.gconvs:
        gconv_block.gconv_1.register_buffer("grid_H", gconv_block.gconv_1.get_kernel_grid_H())

        gconv_block.gconv_1.register_buffer("transformed_grids", gconv_block.gconv_1.transform_kernel_grid_by_group(
            gconv_block.gconv_1.grid_Rn, gconv_block.gconv_1.grid_H
        ))

        gconv_block.gconv_2.register_buffer("grid_H", gconv_block.gconv_2.get_kernel_grid_H())

        gconv_block.gconv_2.register_buffer("transformed_grids", gconv_block.gconv_2.transform_kernel_grid_by_group(
            gconv_block.gconv_2.grid_Rn, gconv_block.gconv_2.grid_H
        ))
