import torch
from matplotlib import pyplot as plt

'''
    Helper functions for batch-processing bilinear interpolation
    https://en.wikipedia.org/wiki/Bilinear_interpolation
'''

def sub2value(values, x_index, y_index):
    '''
        Draw batches of samples from values stored in a 2d matrix, given subscripts
    '''
    batchsize = x_index.size(0)
    width = x_index.size(1)

    # Create linear indices for values tensor
    linear_indices = y_index * values.size(1) + x_index

    # Flatten the values tensor and gather values using linear indices
    flattened_values = values.flatten()
    z_samples = flattened_values[linear_indices]

    # Reshape z_samples to match the desired shape
    z_samples = z_samples.view(batchsize, width)
    return z_samples


def bilinear_interpolation(x_grid, y_grid, values, x, y):
    """
    Perform bilinear interpolation given a regular grid and corresponding values.

    Args:
        x_grid, y_grid (torch.Tensor): 1d array of length = number of grid points, no batch dimension involved
        values (torch.Tensor): Values of the function on the grid.
        x (torch.Tensor): X coordinates for interpolation. dim = batch x width
        y (torch.Tensor): Y coordinates for interpolation.

    Returns:
        torch.Tensor: Interpolated values corresponding to (x, y) coordinates.
    """
 
    # convert continuous coordinates to integer lattice        
    x = (x-x_grid[0])/(x_grid[1]-x_grid[0])
    y = (y-y_grid[0])/(y_grid[1]-y_grid[0])
    

    # find nearest neighbors
    x0 = torch.floor(x).long()  #convert to 'long' integer type (needed for indexing)
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    #clamp value of inputs to borders (use custom backward pass so grad doesn't vanish)
    x0 = torch.clamp(x0, 0, len(x_grid) - 1)
    x1 = torch.clamp(x1, 0, len(x_grid) - 1)
    y0 = torch.clamp(y0, 0, len(y_grid) - 1)
    y1 = torch.clamp(y1, 0, len(y_grid) - 1)

    # Gather the values of the function at the four corners of the interpolation points
    q00 = sub2value(values, y0, x0)
    q01 = sub2value(values, y1, x0)
    q10 = sub2value(values, y0, x1)
    q11 = sub2value(values, y1, x1)
    
    # Calculate weights for bilinear interpolation
    dx1 = x - x0.float()
    dx0 = 1.0 - dx1
    dy1 = y - y0.float()
    dy0 = 1.0 - dy1

    # Perform bilinear interpolation
    interpolated_values = q00 * dx0 * dy0 + q01 * dx0 * dy1 + q10 * dx1 * dy0 + q11 * dx1 * dy1

    return interpolated_values

if __name__=='__main__':
    # Example usage
    # Create an example grid of x and y coordinates
    npoints = 100
    x_grid = torch.linspace(0, torch.pi, npoints)
    y_grid = torch.linspace(0, torch.pi, npoints)

    # generating artificial z_values
    X_mesh,Y_mesh = torch.meshgrid(x_grid,y_grid)
    z_values = torch.cos(X_mesh)+torch.cos(Y_mesh)

    # Example coordinates for interpolation
    batchsize = 2
    width = 3
    x = torch.rand(batchsize, width)*torch.pi
    y = torch.rand(batchsize, width)*torch.pi

    # Interpolate z values
    interpolated_z = bilinear_interpolation(x_grid, y_grid, z_values, x, y)

    print('Interpolated results: ', interpolated_z)
    print('Exact results: ', torch.cos(x)+torch.cos(y))
