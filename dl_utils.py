import torch
import numpy as np
from pyklip.klip import nan_gaussian_filter

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

def rad2arcsec(values):
    """
    Converts the inputs values from radians to arcseconds.

    Parameters
    ----------
    values : Array, radians
        The input values in units of radians to be converted into arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * (3600 * 180) / np.pi


def mask_annulus(im, radius,width=0.3):
    size = im.shape[-1]
    x = np.linspace(-int(size/2) * 0.062424185, int(size/2) * 0.062424185, size)
    y = np.linspace(-int(size/2) * 0.062424185, int(size/2)* 0.062424185, size)
    mesh = np.meshgrid(x,y)
    annulus = np.where((mesh[0]**2 + mesh[1]**2 > (radius-width)**2)&(mesh[0]**2 + mesh[1]**2 < (radius+width)**2), im, np.nan)
    return annulus

def circular_mask(im, peak_x,peak_y, rad_to_mask, value_to_mask, mask_inside=True):
    size = im.shape[-1]
    x = np.linspace(-int(size/2) * 0.062424185, int(size/2) * 0.062424185, size)
    y = np.linspace(-int(size/2) * 0.062424185, int(size/2)* 0.062424185, size)
    mesh = np.meshgrid(x,y)
    if mask_inside:
        circlito = np.where(((mesh[0] - peak_x)**2 + (mesh[1]-peak_y)**2 < rad_to_mask**2), value_to_mask, im)
    else:
        circlito = np.where(((mesh[0] - peak_x)**2 + (mesh[1]-peak_y)**2 > rad_to_mask**2), value_to_mask, im)
    return circlito

# def get_peak_in_circle(im, offset_x,offset_y, rad):
#     origin_x, origin_y = im.shape[0]/2, im.shape[1]/2
#     offset_x_pix, offset_y_pix = offset_x/0.063, offset_y/0.063
#     peakpix_x, peakpix_y = origin_x + offset_x_pix, origin_y+offset_y_pix
#     xcoords = np.arange(im.shape[-1])
#     mesh = np.meshgrid(xcoords,xcoords)
#     circlito = np.where(((mesh[0] - peakpix_x)**2 + (mesh[1]-peakpix_y)**2 < rad**2), im, np.nan)
#     return circlito

def calc_snr(im, offset_x,offset_y, rad_blob, rad_ann, width=0.3, blur_before_signal=False):
    if blur_before_signal:
        im = nan_gaussian_filter(im, 3./2.335)
        positive_blob = circular_mask(im, offset_x,offset_y, rad_blob, np.nan, mask_inside=False)
    else:
        positive_blob = circular_mask(im, offset_x,offset_y, rad_blob, np.nan, mask_inside=False)
        im = nan_gaussian_filter(im, 3./2.335)
    # positive_blob = nan_gaussian_filter(positive_blob, 3)
    # signal = np.nanmax(positive_blob)
    signal = np.nanmax(positive_blob)
    positive_annulus = mask_annulus(im, rad_ann, width)
    # valid_annulus = np.where((~np.isnan(positive_annulus)&(np.isnan(positive_blob))), positive_annulus, np.nan)
    valid_annulus = np.where((~np.isnan(positive_annulus)&(np.isnan(positive_blob))), positive_annulus, np.nan)
    # valid_annulus = nan_gaussian_filter(valid_annulus, 3./2.335)
    # valid_annulus = nan_gaussian_filter(valid_annulus, 3)
    noise = np.nanstd(valid_annulus)
    return signal/noise, valid_annulus, positive_blob