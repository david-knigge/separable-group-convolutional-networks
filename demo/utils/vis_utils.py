import matplotlib.pyplot as plt
import numpy as np

import math


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def visualize_lifting_coordinate_grids(grids):
    """ Visualize the coordinate grid for a lifting convolution kernel
        A lifting coordinate grid has dimension [2, spatial_1, spatial_2]

    :param grids: list of coordinate grids used as input for SIREN
    """

    # determine dimensionality of conv kernel grid
    no_rows = math.floor(math.sqrt(grids.shape[0]))
    no_cols = math.ceil(grids.shape[0] / no_rows)

    fig = plt.figure()
    for grid_idx, grid in enumerate(grids):

        # in case of a lifting layer, we have no_group_elem 2d grids

        ax = fig.add_subplot(no_rows, no_cols, grid_idx + 1)

        x_grid = grid[0].reshape(-1)
        y_grid = grid[1].reshape(-1)

        ax.scatter(x_grid[0], y_grid[0], marker="x", s=128)
        ax.scatter(x_grid[1:], y_grid[1:], c='#FFA8A8')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                       left=False, labelleft=False)
        # ax.set_title(f"grid transformed for $g_{grid_idx}$")

def visualize_group_coordinate_grids(grids):
    """ Visualize the coordinate grid for a groups convolution kernel.
        A groups coordinate grid has dimension [3, num_group_elems, spatial_1, spatial_2]

    :param grids: list of coordinate grids used as input for SIREN
    """
    no_rows = math.floor(math.sqrt(grids.shape[0]))
    no_cols = math.ceil(grids.shape[0] / no_rows)

    fig = plt.figure()

    for grid_idx, grid in enumerate(grids):
        ax = fig.add_subplot(no_rows, no_cols, grid_idx + 1, projection="3d")

        x_grid = grid[0]
        y_grid = grid[1]
        z_grid = grid[2]

        no_group_elems = grid.shape[-3]
        cmap = get_cmap(no_group_elems + 1)

        for group_elem_idx in range(no_group_elems):
            ax.scatter(
                x_grid[group_elem_idx, 0, 0],
                y_grid[group_elem_idx, 0, 0],
                z_grid[group_elem_idx, 0, 0],
                marker="x",
                s=128,
                color=cmap(group_elem_idx)
            )

            ax.scatter(
                x_grid[group_elem_idx].reshape(-1)[1:],
                y_grid[group_elem_idx].reshape(-1)[1:],
                z_grid[group_elem_idx].reshape(-1)[1:],
                color=cmap(group_elem_idx)
            )

            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                           left=False, labelleft=False)

        ax.set_zlabel("groups dim")
        ax.set_title(f"grid transformed for $g_{grid_idx}$")


def visualize_lifting_kernels(kernels, save_path=''):
    """

    :param kernels:
    :return:
    """
    no_group_elements_out = len(kernels)
    no_out_channels = kernels[0].shape[0]
    no_in_channels = kernels[0].shape[1]

    for out_channel in range(no_out_channels):

        fig = plt.figure()
        fig_idx = 0
        for in_channel in range(no_in_channels):

            for group_elem_out in range(no_group_elements_out):

                filt = kernels[group_elem_out][out_channel, in_channel, :, :]
                ax = fig.add_subplot(no_in_channels, no_group_elements_out, fig_idx + 1)
                img = ax.imshow(filt)
                # fig.colorbar(img)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                left=False, labelleft=False)
                # ax.set_title(f"$C^{in_channel}_{out_channel} g_{group_elem_out}$")
                fig_idx += 1

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    if save_path:
        plt.savefig(save_path + f'{out_channel}' + '.png')
    else:
        plt.show()


def visualize_group_kernels(kernels, save_path=None):
    no_group_elements_out = len(kernels)
    no_out_channels = kernels[0].shape[0]
    no_in_channels = kernels[0].shape[1]
    kernel_size = kernels[0].shape[-1]

    no_out_channels = kernels.shape[0]
    no_in_channels = kernels.shape[2]

    no_group_elements_out = kernels.shape[1]
    no_group_elements_in = kernels.shape[3]

    for out_channel in range(no_out_channels):

        fig = plt.figure(figsize=(10, no_in_channels * 10))
        fig_idx = 0
        for in_channel in range(no_in_channels):

            for group_elem_out in range(no_group_elements_out):

                # filter containing input group elements and input spatial dimensions
                filt = kernels[out_channel, group_elem_out, in_channel, :, :, :]

                # concatenated filter of input group elements
                catfilt = np.concatenate([f for f in filt], axis=0)

                ax = fig.add_subplot(no_in_channels, no_group_elements_out, fig_idx + 1)
                img = ax.imshow(catfilt)
                # fig.colorbar(img)

                # ax.set_title(f"$C^{in_channel}_{out_channel} g_{group_elem_out}$")
                ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                left=False, labelleft=False)
                fig_idx += 1

        if save_path:
            plt.savefig(save_path + f'{out_channel}' + '.png')
        else:
            plt.show()


def visualize_activations(acts, channel_idx=0, sample_idx=0):
    """ Visualize the activations throughout the network.

    :param acts: tensor of activations with shape [batch_dim, out_channels, group_dim, spatial_1, spatial_2]
    :param channel_idx: integer, determines which of the output channels to visualize over the groups
    :param sample_idx: integer, determines which of the samples to visualize for
    :return: None
    """
    act = acts[sample_idx, channel_idx, :, :, :]

    no_rows = math.floor(len(act) // 2)
    no_cols = math.ceil(len(act) / no_rows)

    fig = plt.figure()

    for group_elem in range(len(act)):
        ax = fig.add_subplot(no_rows, no_cols, group_elem + 1)

        img = ax.imshow(act[group_elem, :, :])
        fig.colorbar(img)

        ax.set_title(f"Activations for channel {channel_idx}, $g_{group_elem}$")

    plt.show()
