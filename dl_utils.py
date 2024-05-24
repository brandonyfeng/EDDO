import torch
import numpy as np

def calc_nfringes(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: int,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    focal_shift: float = 0.0,
):
    """
    Calculates the number of fringes in the output plane.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.

    Returns
    -------
    nfringes : Array
        The number of fringes in the output plane.
    """
    # Fringe size
    diameter = npixels_in * pixel_scale_in
    fringe_size = wavelength / diameter

    # Output array size
    output_size = npixels_out * pixel_scale_out
    if focal_length is not None:
        output_size /= focal_length + focal_shift

    # Fringe size and number of fringes
    return output_size / fringe_size



def nd_coords(npixels, pixel_scales = 1.0, offsets=0.0, indexing: str = "xy",):

    if not isinstance(npixels, tuple):
        npixels = (npixels,)
    if not isinstance(pixel_scales, tuple):
        pixel_scales = (pixel_scales,) * len(npixels)
    if not isinstance(offsets, tuple):
        offsets = (offsets,) * len(npixels)

    def pixel_fn(n, offset, scale):
        pix = torch.arange(n) - (n - 1) / 2.0
        pix *= scale
        pix -= offset
        return pix

    lin_pixels = []
    for n, o, p in zip(npixels, offsets, pixel_scales):
      lin_pixels += [pixel_fn(n, o, p)]

    positions = torch.from_numpy(np.array(np.meshgrid(*lin_pixels, indexing=indexing)))
    positions = positions.to(dtype=torch.float64)

    return torch.squeeze(positions)


def transfer_matrix(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    shift: float = 0.0,
    focal_length: float = None,
    focal_shift: float = 0.0,
    inverse: bool = False,
):
    """
    Calculates the transfer matrix for the MFT.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    shift : float = 0.0
        The shift in the center of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.
    inverse: bool = False
        Is this a forward or inverse propagation.

    Returns
    -------
    transfer_matrix : Array
        The transfer matrix for the MFT.
    """
    # Get parameters
    fringe_size = wavelength / (pixel_scale_in * npixels_in)

    # Input coordinates
    scale_in = 1.0 / npixels_in
    in_vec = nd_coords(npixels_in, scale_in, shift * scale_in)

    # Output coordinates
    scale_out = pixel_scale_out / fringe_size
    if focal_length is not None:
        # scale_out /= focal_length
        scale_out /= focal_length + focal_shift
    out_vec = nd_coords(npixels_out, scale_out, shift * scale_out)

    # Generate transfer matrix
    matrix = 2j * np.pi * torch.outer(in_vec, out_vec)
    if inverse:
        matrix *= -1
        
    return torch.exp(matrix)


def dl_MFT(
    phasor,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift = [0.0, 0.0],
    pixel: bool = True,
    inverse: bool = False,
):
    """
    Propagates a phasor using a Matrix Fourier Transform (MFT), allowing for output
    pixel scale and a shift to be specified.

    TODO: Add link to Soumer et al. 2007(?), which describes the MFT.

    Parameters
    ----------
    phasor : Array
        The input phasor.
    wavelength : float, meters
        The wavelength of the input phasor.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    shift : Array = np.zeros(2)
        The shift in the center of the output plane.
    pixel : bool = True
        Should the shift be taken in units of pixels, or pixel scale.


    Returns
    -------
    phasor : Array
        The propagated phasor.
    """
    # Get parameters
    npixels_in = phasor.shape[-1]
    if not pixel:
        shift /= pixel_scale_out

    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )
    x_mat = get_tf_mat(shift[0]).to(phasor.device)
    y_mat = get_tf_mat(shift[1]).to(phasor.device)
    phasor = (y_mat.T @ phasor) @ x_mat

    # Normalise
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    phasor *= np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return phasor


def partial_MFT(
    npixels_in,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift = [0.0, 0.0],
    pixel: bool = True,
    inverse: bool = False,
):
    if not pixel:
        shift /= pixel_scale_out

    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )
    x_mat = get_tf_mat(shift[0])
    y_mat = get_tf_mat(shift[1])
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    mult = np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return x_mat, y_mat, mult



def pixel_coords(npixels: int, diameter: float):
    coords = nd_coords((npixels,) * 2, (diameter / npixels,) * 2)
    #coords = torch.flip(coords, [1])
    return coords

def crop_to(array, npixels: int):
    npixels_in = array.shape[-1]
    start, stop = (npixels_in - npixels) // 2, (npixels_in + npixels) // 2
    return array[..., start:stop, start:stop]


def cart2polar(coordinates):
    x, y = coordinates
    return torch.array([torch.hypot(x, y), torch.arctan2(y, x)])

def circle(coords, radius):
    return torch.where(coords[0]**2 + coords[1]**2 < radius**2, 1.0, 0.0)


def arcsec2rad(values):
    """
    Converts the inputs values from arcseconds to radians.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into radians.

    Returns
    -------
    values : Array, radians
        The input values converted into radians.
    """
    return values * np.pi / (3600 * 180)

