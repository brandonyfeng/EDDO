import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import math

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import copy
import pandas as pd
import glob

from dl_utils import pixel_coords, crop_to, arcsec2rad, partial_MFT,calc_snr


def load_data(data_dir, num_wl=None, oversample=None):
    basic_masks_dir = os.path.join(data_dir, 'masks_1024')

    aperture = np.load(f'{basic_masks_dir}/primary_transmission_1024.npy')
    aperture = np.flip(aperture, axis=0)
    lyot = np.load(f'{basic_masks_dir}/circlyotstop_transmission_1024.npy')
    if num_wl is not None or oversample is not None:
        data_dir = os.path.join(data_dir, f'npix_{int(1024*oversample)}_num_wl_{int(num_wl)}')
    # else:
        
        # basic_masks_dir = os.path.join(data_dir, f'mask_{int(1024*oversample)}')

        # aperture = np.load(f'{basic_masks_dir}/primary_transmission_1024.npy')
        # aperture = np.flip(aperture, axis=0)
        # lyot = np.load(f'{basic_masks_dir}/circlyotstop_transmission_1024.npy')

    # WAVELENGTH DEPENDENCE

    wlens_weights = np.load(os.path.join(data_dir, 'wlens_weights', 'lambda_weights.npy'))
    fpm, nircam_opd = [], []
    for i in range(len(wlens_weights[0])):
        currfpm = np.load(f'{data_dir}/mask335r_transmissions/mask335r_transmission_{i}.npy')
        curr_nircamopd = np.load(f'{data_dir}/fov_wl_nircam_opds/fov_wl_nircam_opd_{i}.npy')
        fpm.append(currfpm)
        nircam_opd.append(curr_nircamopd)

    fpm = np.array(fpm)
    nircam_opd = np.array(nircam_opd)

    return aperture, lyot, fpm, nircam_opd, wlens_weights

def weighted_smooth_l1_loss(pred, obs, noise, beta=1.0, reduction='mean'):
    """
    Computes a noise-weighted Smooth L1 Loss.
    
    Args:
        pred: Predicted tensor of shape (1, H, W)
        obs: Observed tensor of shape (1, H, W)
        noise: Per-pixel noise (stddev), shape (1, H, W)
        beta: Transition point from L1 to L2 in smooth L1 loss
        reduction: 'mean', 'sum', or 'none'
    """
    # Compute per-pixel smooth L1 loss
    diff = pred - obs
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    
    # Weight by inverse variance (1 / noise^2)
    weights = 1.0 / (noise ** 2 + 1e-8)  # avoid division by zero
    weighted_loss = loss * weights
    
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss

def weighted_l1_loss(pred, obs, noise, reduction='mean'):
    """
    Computes a noise-weighted L1 loss, assuming Laplace-distributed noise.
    
    Args:
        pred: Predicted tensor of shape (1, H, W)
        obs: Observed tensor of shape (1, H, W)
        noise: Per-pixel noise (stddev), shape (1, H, W)
        reduction: 'mean', 'sum', or 'none'
    """
    weights = 1.0 / (noise + 1e-8)  # avoid division by zero
    loss = weights * (pred - obs).abs()
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def load_mirror_segment_info(data_dir):
    segments_folder = os.path.join(data_dir, 'mirror_segments')

    segments_masks = np.load(os.path.join(segments_folder, 'segment_masks_1024.npy'))

    with open(os.path.join(segments_folder, 'seg_centers_pixels_1024.json'), 'r') as file:
        segment_centers_pixels = json.load(file)
    
    with open(os.path.join(segments_folder, 'segnames_idx.json'), 'r') as file:
        segment_idx = json.load(file)

    return segments_masks, segment_centers_pixels, segment_idx

def total_variation_loss(img):
    # Compute the total variation loss, for smoothness
    # horizontal_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    # vertical_diff = img[:, 1:, :, :] - img[:, :-1, :, :]
    # horizontal_diff = img[:, 1:, :] - img[:, :-1, :]
    # vertical_diff = img[1:, :, :] - img[:-1, :, :]
    # TEST TO SEE IF THIS MAKES SENSE
    horizontal_diff = img[:, 1:, :] - img[:, :-1, :]
    vertical_diff = img[:, :, 1:] - img[:, :, :-1]
    tv_loss = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    return tv_loss

def shift_image_subpixel(image: torch.Tensor, shift_x: float = -0.3, shift_y: float = 0.1) -> torch.Tensor:
    """
    Shift an image by sub-pixel amounts using bilinear interpolation.
    
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W) or (N, C, H, W)
        shift_x (float): Amount to shift in the x-direction (positive = right)
        shift_y (float): Amount to shift in the y-direction (positive = down)

    Returns:
        torch.Tensor: Shifted image of same shape
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    N, C, H, W = image.shape

    # Create normalized 2D affine matrix for translation
    theta = torch.tensor([[
        [1, 0, -2 * shift_x / W],  # x shift
        [0, 1, -2 * shift_y / H]   # y shift
    ]], dtype=torch.double, device=image.device)

    # Repeat theta for batch
    theta = theta.expand(N, -1, -1)

    # Create grid and sample
    grid = F.affine_grid(theta, size=image.size(), align_corners=False)
    shifted_image = F.grid_sample(image, grid, mode='bicubic', padding_mode='border', align_corners=False)

    return shifted_image.squeeze(0) if shifted_image.size(0) == 1 else shifted_image

def gaussian_log_prior(x, mu, sigma):
    """Negative log-likelihood of Gaussian prior."""
    return ((x - mu)**2).sum() / (2 * sigma**2)

class ShiftModule(nn.Module):
    def __init__(self, height, width, learn_scale=False):
        super(ShiftModule, self).__init__()
        self.x_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.y_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((x_coords, y_coords), dim=-1).float()
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / (width - 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / (height - 1) - 1.0
        self.grid = nn.Parameter(grid[None], requires_grad=False)
        self.rotation = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=learn_scale)

    def forward(self, x):
        grid = self.grid.clone().to(x.dtype) * self.scale

        cos_theta = torch.cos(self.rotation)
        sin_theta = torch.sin(self.rotation)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]]).to(x.device).to(x.dtype)
        grid = torch.matmul(grid.view(-1, 2), rotation_matrix).view(1, x.shape[-2], x.shape[-1], 2)

        grid[..., 0] = grid[..., 0] + self.x_shift
        grid[..., 1] = grid[..., 1] + self.y_shift

        out = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='reflection', align_corners=False)

        return out[0]


class GridOffsetModule(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        grid = torch.zeros((height, width))
        self.grid = nn.Parameter(grid[None], requires_grad=True)

    def get_res(self):
        res = self.grid
        return res

    def forward(self, x=None):
        res = self.grid
        if x is None:
            return res
        out = res + x
        return out


class OPDOffsetModule(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        grid = torch.zeros((1, height, width))
        self.grid = nn.Parameter(grid, requires_grad=True)

    def get_res(self):
        res = self.grid
        return res

    def total_variation_loss(self):
        # Scale the grid by its mean
        scaled_grid = self.grid / (torch.mean(torch.abs(self.grid)) + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate differences in x and y directions
        diff_x = torch.abs(scaled_grid[:, :, 1:] - scaled_grid[:, :, :-1])
        diff_y = torch.abs(scaled_grid[:, 1:, :] - scaled_grid[:, :-1, :])
        
        # Sum up the differences
        tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
        
        return tv_loss

    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out

class FluxOffsetModule(nn.Module):
    def __init__(self, flux=None):
        super().__init__()
        # self.data = nn.Parameter(torch.ones(1), requires_grad=True)
        if flux is None:
            self.data = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.data = nn.Parameter(flux, requires_grad=True)

    def forward(self, x):
        return self.data * x

# class MultiplyLayer(nn.Module):
#     def __init__(self, alpha_init=1.0):  # Initialize with a default value
#         super(MultiplyLayer, self).__init__()
#         self.alpha = nn.Parameter(torch.tensor(alpha_init))  # Create a learnable parameter

#     def forward(self, x):
#         return x * self.alpha

class AngleOffsetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self):

        return self.data

class LinearInterpOPD(nn.Module):
    """
    Linear interpolation between two OPD measurements
    """
    def __init__(self, before_opd, after_opd, coeff_init=0.):
        super().__init__()
        self.coeff = nn.Parameter(torch.FloatTensor([coeff_init]), requires_grad=True)
        self.before_opd = before_opd
        self.after_opd = after_opd

    def get_res(self):
        return (self.after_opd - self.before_opd) * self.coeff
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out

class PTT_OPD(nn.Module):
    """
    Parametrizes the OPD at the entrance of JWST as Piston-Tip-Tilt of the hexagonal mirror segments
    """
    def __init__(self,labelled_transmission,segment_centers, segment_labels,npixels):
        super().__init__()
        self.piston_coeffs = nn.Parameter(torch.zeros(18))
        self.xtilt_coeffs = nn.Parameter(torch.zeros(18))
        self.ytilt_coeffs = nn.Parameter(torch.zeros(18))
        # Origin 0,0 at the corner, to match the origin of the segment_centers keyword.
        # Only used to shift the origin of the Zernikes to the center of each segment
        coords = np.meshgrid(np.linspace(0.,npixels, npixels), np.linspace(0.,npixels, npixels))
        self.coords_grid_x = torch.from_numpy(coords[0]).to(DEVICE)[None]
        self.coords_grid_y = torch.from_numpy(coords[1]).to(DEVICE)[None]
        self.segment_radius = nn.Parameter((npixels / 5. / 2.) / torch.cos(torch.deg2rad(torch.tensor(30.))), requires_grad=False) # in pixels

        ordered_segment_masks = []
        ordered_segment_centers = []

        for seg_name, seg_label in segment_labels.items():
            current_mirror_transmission = np.where(labelled_transmission == seg_label, 1.,0.)
            # Flip to match orientation of our simulator
            ordered_segment_masks.append(torch.from_numpy(np.flip(current_mirror_transmission, axis=0).copy())[None])

            # Invert y-axis of center coordinates, to match orientation of our simulator
            center_x = segment_centers[seg_name][0]
            center_y =npixels - segment_centers[seg_name][1]
            ordered_segment_centers.append([center_x, center_y])
        
        self.ordered_segment_masks = torch.stack(ordered_segment_masks).to(DEVICE)
        self.ordered_segment_centers = torch.tensor(ordered_segment_centers, dtype=torch.float64).to(DEVICE)

    def get_res(self):
        total_opd = torch.zeros_like(self.coords_grid_x)
        for i in range(len(self.ordered_segment_masks)):
            piston_opd = torch.zeros_like(self.ordered_segment_masks[i])+ self.piston_coeffs[i] # simply add constant 
            r_grid = torch.sqrt((self.coords_grid_x - self.ordered_segment_centers[i][0])**2 + (self.coords_grid_y-self.ordered_segment_centers[i][1])**2)
            theta_grid = torch.arctan2(self.coords_grid_y-self.ordered_segment_centers[i][1], self.coords_grid_x- self.ordered_segment_centers[i][0])

            xtilt_opd = self.xtilt_coeffs[i] * 2* r_grid * torch.cos(theta_grid) / self.segment_radius # Normalize to unit radius
            ytilt_opd = self.ytilt_coeffs[i] * 2* r_grid * torch.sin(theta_grid) / self.segment_radius

            total_opd+=(piston_opd + xtilt_opd + ytilt_opd)* self.ordered_segment_masks[i]

        return total_opd
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out


class Wavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: float, peak_flux: float, angles = None):
        super().__init__()
        self.wavelength = nn.Parameter(wavelength, requires_grad=False)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumber = 2 * np.pi / self.wavelength
        self.npixels = npixels
        self.diameter = diameter
        self.peak_flux = nn.Parameter(peak_flux, requires_grad=True)
        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles)
        self.reset()

    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasor(self, angles_offset=None):
        opd = self.get_tilt_opd(angles_offset)
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilt_opd(self, angles_offset=None):
        if angles_offset is not None:
            opd = -((self.angles + angles_offset)[:, None, None] * self.coordinates).sum(0)
        else:
            opd = -(self.angles[:, None, None] * self.coordinates).sum(0)

        opd = opd[None] * self.wavenumber
        return opd

    def tilt(self, angles):
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        coords = self.coordinates
        opd = -(angles[:, None, None] * coords).sum(0)
        opd = opd[None]
        return self.add_opd(opd)

    def add_opd(self, opd):
        self.phase.data = self.phase.data + self.wavenumber * opd

    def propagate(self, phasor = None, pad: int = 2):
        npixels = self.npixels

        if pad > 1:
            _npixels = (npixels * (pad - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasor = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])

        return phasor

    def forward(self, phasor, layer, normalize=False):
        new_phasor = phasor * layer
        if normalize:
            denom = new_phasor.abs() ** 2
            denom = torch.sum(denom, dim=(1, 2), keepdim=True) ** 0.5
        else:
            denom = 1.
        return new_phasor / denom

    def forward_fpm(self, phasor, layer, oversample=1):
        npixels_in = self.npixels
        phasor = self.propagate(phasor, pad=oversample)
        new_phasor = phasor * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfe(self, phasor, layer, wlen):
        wfe = torch.exp(1j * layer * 2 * np.pi / wlen)
        new_phasor = phasor * wfe
        return new_phasor

class LearnableGaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, init_sigma=0.28):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))  # Learn in log-space

    def forward(self, x):
        # x: (1, H, W)
        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError("Input must have shape (1, H, W)")

        sigma = torch.exp(self.log_sigma)
        coords = torch.arange(self.kernel_size, dtype=torch.double, device=x.device) - self.kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, K, K)

        # padding = self.kernel_size // 2
        x_reshaped = x.unsqueeze(0)  # Shape: (1, 1, H, W)

        blurred = F.conv2d(x_reshaped, kernel_2d, padding='same')
        return blurred.squeeze(0)  # Shape: (1, H, W)

class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, args,oversample=None, use_ptt = None,OTE_wfe_basis=None,second_delta_wfe=None, num_det_px = 80):
        super().__init__()
        self.aperture = aperture
        self.lyot = lyot
        self.fpm = fpm
        self.nircam_opd = nircam_opd
        self.num_det_px = num_det_px
        if oversample is None:
            oversample = 1
        self.oversample = oversample

        self.lyot_shifts = ShiftModule(lyot.shape[-2], lyot.shape[-1])
        # self.lyot_shifts.x_shift = nn.Parameter(torch.FloatTensor([100.]), requires_grad=True)
        # self.lyot_shifts.y_shift = nn.Parameter(torch.FloatTensor([100.]), requires_grad=True)
        self.fpm_shifts = ShiftModule(fpm.shape[-2], fpm.shape[-1])
        self.nircam_offsets = GridOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])

        self.flux_correction = FluxOffsetModule()

        if self.oversample!=1:
            kernel_sigma = 0.28 * self.oversample
            # kernel_sigma = 0.27979042973779844 * self.oversample
            kernel_size = round(4.0 * kernel_sigma)

            self.charge_diffusion = LearnableGaussianBlur(kernel_size=kernel_size, init_sigma=kernel_sigma)
        else:
            self.charge_diffusion = LearnableGaussianBlur()

        # if use_ptt is None:
        #     self.wfe_offsets = OPDOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])
        # else:
        #     self.wfe_offsets = use_ptt
        self.wfe_offsets = OTE_wfe_basis
        self.second_wfe_offsets = second_delta_wfe
        self.angle_offsets = AngleOffsetModule()

        (npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse) = args
        xmats, ymats, mults = [], [], []
        for i in range(len(wavelengths)):
            args = (npixels, wavelengths[i], true_pixel_scale, psf_npix*self.oversample, psf_pixel_scale/self.oversample, focal_length, shift, pixel, inverse)
            x_mat, y_mat, mult = partial_MFT(*args)
            xmats.append(x_mat); ymats.append(y_mat); mults.append(torch.tensor(mult, dtype=torch.float64))
        self.x_mat = nn.Parameter(torch.stack(xmats), requires_grad=False)
        self.y_mat = nn.Parameter(torch.stack(ymats), requires_grad=False)
        self.mult = nn.Parameter(torch.stack(mults), requires_grad=False)


    def forward(self, wavefront_list, wfe, wl_weights, wavelenghts, save_wf=False):
        output = None
        if save_wf:
            wftoreturn = []
        wfe_ = self.wfe_offsets(wfe)
        if self.second_wfe_offsets is not None:
            wfe_ = self.second_wfe_offsets(wfe_)
        for i in range(len(wl_weights)):
            wavefront = wavefront_list[i]
            phasor = wavefront.get_phasor(self.angle_offsets())
            phasor = wavefront.forward_wfe(phasor, wfe_, wavelenghts[i])
            phasor_ap = wavefront.forward(phasor, self.aperture, normalize=True)
            phasor_fpm = wavefront.forward_fpm(phasor_ap, self.fpm_shifts(self.fpm[:, i]),oversample=self.oversample)
            # phasor_fpm = phasor_ap
            phasor_lyot = wavefront.forward(phasor_fpm, self.lyot_shifts(self.lyot))
            # phasor_lyot = phasor_fpm
            phasor_nircam_opd = wavefront.forward_wfe(phasor_lyot, self.nircam_offsets(self.nircam_opd[:, i]), wavelenghts[i])
            if save_wf and i ==0:
                wftoreturn.append(torch.clone(phasor_nircam_opd))
            phasor = (self.y_mat[i].T @ phasor_nircam_opd) @ self.x_mat[i]
            phasor *= self.mult[i]
            # w = (wavefront.peak_flux) ** 0.5
            w = (wavefront.peak_flux) ** 0.5
            out = (torch.abs(phasor) * w) ** 2 
            # out = (torch.abs(phasor) * self.flux_correction(0.)**0.5) ** 2 
            if output is None:
                output = out * wl_weights[i]
            else:
                output += out * wl_weights[i]
        
        # print('OUTPUT HERE HAS SHAPE ', output.shape)
        output = self.charge_diffusion(output)
        # print('OUTPUT HERE AFTER HAS SHAPE ', output.shape)
        if self.oversample != 1:
            # Rebin while conserving flux
            output = torch.sum(torch.reshape(output, (1,self.num_det_px,self.oversample,self.num_det_px,self.oversample)), (2,4))

        # output = torch.flip(output, dims=(-2,))

        # output = torch.mul(output,self.flux_correction())

        output = self.flux_correction(output)

        # TRY SHIFTING HERE

        output = shift_image_subpixel(output)

        # Valid for NIRCam; right sigma probably depends on filter/detector/etc
        # Charge diffusion, probably the most significant detector effect at play here
        # Other detector effects can be added as convolutions with the kernels in the data/detector_kernels folder
        # output = v2.GaussianBlur(kernel_size=3, sigma=0.28)(output)
        if save_wf:
            return output, wftoreturn
        else:
            return output

    def forward_val(self, wavefront):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward(phasor, self.aperture)
        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.peak_flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out

## ZERNIKE POLYNOMIALS PARAMETRIZATION
class ZernikeBasis:
    def __init__(self, height, width, max_modes):
        self.H = height
        self.W = width
        self.N = max_modes
        self.r, self.theta = self._create_polar_grid()
        self.mask = self.r <= 1.0
        self.basis = self._generate_basis()

    def _create_polar_grid(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, self.H),
                              torch.linspace(-1, 1, self.W),
                              indexing='ij')
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        r[r > 1] = 0  # Mask outside the unit disk
        return r, theta

    def _zernike_radial(self, n, m, r):
        R = torch.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coeff = ((-1)**k * math.comb(n - k, k) * math.comb(n - 2*k, (n - abs(m)) // 2 - k))
            R += coeff * r**(n - 2*k)
        return R

    def _zernike_polynomial(self, n, m):
        R = self._zernike_radial(n, m, self.r)
        if m > 0:
            return R * torch.cos(m * self.theta)
        elif m < 0:
            return R * torch.sin(-m * self.theta)
        else:
            return R

    def _generate_basis(self):
        basis = []
        count = 0
        n = 0
        while count < self.N:
            for m in range(-n, n + 1, 2):
                basis.append(self._zernike_polynomial(n, m))
                count += 1
                if count >= self.N:
                    break
            n += 1
        return torch.stack(basis, dim=0)  # Shape: (N, H, W)

    def get_basis(self):
        return self.basis.clone()

    def get_mask(self):
        return self.mask.clone()

class ZernikeModel(nn.Module):
    def __init__(self, zernike_basis: ZernikeBasis):
        super().__init__()
        self.basis = zernike_basis.get_basis().to(DEVICE)[None]  # (1,N, H, W)
        self.mask = zernike_basis.get_mask().to(DEVICE)[None]
        self.coeffs = nn.Parameter(torch.zeros(1, self.basis.shape[1], device=DEVICE), requires_grad=True) # Learnable coefficients

    def get_res(self):
        wf = torch.sum(self.coeffs[:,:,None,None] * self.basis, dim=1)* self.mask 
        # print('---- req grad???', self.coeffs.requires_grad)
        return wf  # Mask outside unit disk

    def total_variation_loss(self):
        # Scale the grid by its mean
        scaled_grid = self.grid / (torch.mean(torch.abs(self.grid)) + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate differences in x and y directions
        diff_x = torch.abs(scaled_grid[:, :, 1:] - scaled_grid[:, :, :-1])
        diff_y = torch.abs(scaled_grid[:, 1:, :] - scaled_grid[:, :-1, :])
        
        # Sum up the differences
        tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
        
        return tv_loss

    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out

class IECModesModelOPD(nn.Module):
    """
    Uses as basis the PCA'd modes from IEC testing (Telfer 2024), computed from data shared by Laurent Pueyo.
    """
    def __init__(self, max_modes):
        super().__init__()
        # self.coeffs =nn.Parameter(torch.zeros(max_modes), requires_grad=True)
        self.basis = self._load_basis(max_modes).to(DEVICE)[None]
        self.coeffs = nn.Parameter(torch.zeros(1, self.basis.shape[1], device=DEVICE), requires_grad=True)

    def _load_basis(self, max_modes):
        modes_numpy = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/SPIE_modes_analysis/top200IECmodes_1024.npy')
        return torch.from_numpy(modes_numpy[:max_modes]) # Shape (N, H, W)

    def get_res(self):
        wf = torch.sum(self.coeffs[:,:,None,None] * self.basis, dim=1)
        # print('---- req grad???', self.coeffs.requires_grad)
        return wf  # Mask outside unit disk
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out    
    
class OTEMeasModesModelOPD(nn.Module):
    """
    Uses as basis the PCA'd modes from IEC testing (Telfer 2024), computed from data shared by Laurent Pueyo.
    """
    def __init__(self, max_modes):
        super().__init__()
        # self.coeffs =nn.Parameter(torch.zeros(max_modes), requires_grad=True)
        self.basis = self._load_basis(max_modes).to(DEVICE)[None]
        self.coeffs = nn.Parameter(torch.zeros(1, self.basis.shape[1], device=DEVICE), requires_grad=True)

    def _load_basis(self, max_modes):
        modes_numpy = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/SPIE_modes_analysis/top100primaryOPD_modes_befsecondimpact_normalized1rms.npy')
        return torch.from_numpy(modes_numpy[:max_modes]) # Shape (N, H, W)

    def get_res(self):
        wf = torch.sum(self.coeffs[:,:,None,None] * self.basis, dim=1)
        # print('---- req grad???', self.coeffs.requires_grad)
        return wf  # Mask outside unit disk
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out   

def get_date_from_obs(curr_obs_path, obs_info_df, program_no):
    # First get the observation number and dither number
    obsname = os.path.basename(curr_obs_path)
    if obsname[:3] == 'ref':
        obsnum = obsname[10:13]
        dithnum = obsname[-9:-4]
    else:
        obsnum = obsname[7:10]
        dithnum = obsname[-9:-4]

    # Assume that only one match is found, which is reasonable????
    date_obs_res = obs_info_df['DATE-OBS'][(obs_info_df['FILENAME'].str.contains(obsnum+'001')) & (obs_info_df['FILENAME'].str.contains(program_no)) & (obs_info_df['FILENAME'].str.contains(dithnum))].item()
    return date_obs_res, obsnum, dithnum

def get_before_and_after_opd(date_obs_string):
    allOPDs = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/SPIE_modes_analysis/allOPDssaved/processedOPDs.npy')
    opds_meas_info = pd.read_csv('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/SPIE_modes_analysis/allOPDssaved/filenames_OPDs.csv')

    dates_before = opds_meas_info[opds_meas_info['DATE-OBS'] <= date_obs_string]
    closest_date_before = dates_before['DATE-OBS'].max()
    correct_index = dates_before[dates_before['DATE-OBS'] == closest_date_before]['Index'].item()

    return [allOPDs[correct_index], allOPDs[correct_index+1]]

# PUT EVERYTHING INTO THIS FUNCTION
def run_optimization(data_dir, root_dir, scene_name,exp_name, star_offset_x, star_offset_y, use_ptt,reference_file, measurement_file,lr,iters, vis_freq,ref_cutoff_iter, num_wl, oversample,num_det_px,sci_targ_name,num_ints,ref_which_int, sci_which_int,use_linear_interp_opd,no_median,measurement_noisemap,reference_noisemap,use_simulated_data,px_mask_file,fit_Lyot,fit_everything,stage_fluxpos_cutoff_iter,smooth,OPD_loss_weight,freeze_position,be_normal,sci_lr_weight,fit_second_wfe_offsets, primaryOPD_basis,other_OPD_meas,big_lr_factor,freeze_OTE_OPD_sci,insert_initial_delta_OTE_OPD,insert_initial_delta_NIRCam_OPD,correct_OPD_array_as_loaded=None, injected_companion=None, pre_initialized_state=None):
    wf_npix = 1024
    # Diameter of the aperture
    diameter = 6.603464
    # Number of pixels in the PSF
    psf_npix = num_det_px
    # Pixel scale in our detector, in arcseconds. For our case, around 14 miliarcseconds per pixel is expected.
    psf_pixel_scale = 0.062424185

    # Load data from args.data_dir
    if oversample is None and num_wl is None:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=data_dir)
    else:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=data_dir, num_wl=num_wl, oversample=oversample)
        
     
    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]

    if insert_initial_delta_NIRCam_OPD:
        new_opd_array = []
        delta_NIRCam_OPD = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/new4050_files/OPD_systematics/mean_delta_NIRCam_OPD_group2_4.npy')
        for element in nircam_OPD:
            new_opd_array.append(element+delta_NIRCam_OPD)
        nircam_OPD = np.array(new_opd_array).copy()
        
    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights)

    # correct_OPD_array_as_loaded must a list or tuple with two elements: the before OPD and after OPD, as computed from the function upstream
    if correct_OPD_array_as_loaded is None:
        sampledWFEs = np.load(f'{data_dir}/masks_2048/observation_opd.npy')
        sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
    else:
        sampledWFEs = np.flip(correct_OPD_array_as_loaded[0], axis=0)[None]

    if insert_initial_delta_OTE_OPD:
        delta_OTE_OPD = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/new4050_files/OPD_systematics/mean_delta_OTE_OPD_group2_4.npy')
        delta_OTE_OPD = delta_OTE_OPD[None]
        sampledWFEs +=delta_OTE_OPD.copy()

    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch = sampledWFEs.contiguous().to(DEVICE)

    if correct_OPD_array_as_loaded is None:
        if sci_targ_name is None:
            sampledWFEs_after = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/4050_data_for_diff_modeling/group_2_after/masks_2048/observation_opd.npy')
        elif sci_targ_name == 'HR8799':
            sampledWFEs_after = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HR8799stuff/after_data/masks_2048/observation_opd.npy')
        elif args.sci_targ_name == 'HIP65426':
            sampledWFEs_after = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/afterOPD/masks_2048/observation_opd.npy')
        else:
            sampledWFEs_after = np.load(other_OPD_meas)
        
        sampledWFEs_after = np.flip(sampledWFEs_after, axis=0)[None]
    else:
        sampledWFEs_after = np.flip(correct_OPD_array_as_loaded[1], axis=0)[None]

    if insert_initial_delta_OTE_OPD:
        sampledWFEs_after +=delta_OTE_OPD.copy()
    sampledWFEs_after = torch.from_numpy(sampledWFEs_after.copy()).float()
    sampledWFEs_after = F.interpolate(sampledWFEs_after[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch_after = sampledWFEs_after.contiguous().to(DEVICE)


    wfe_batch_list = [wfe_batch, wfe_batch_after]
    ############
    # Set up export directories
    vis_dir = f'{root_dir}/vis/{scene_name}/{exp_name}/initial_fit_reference'
    os.makedirs(vis_dir, exist_ok=True)

    contrast_normalization = 0.003716380479003919
    sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data; VERY ROUGH! To be fixed
    photons_normalization = 90578.00102527262 * sim_to_real_scaling *1e4 / 2251.24456327477#THREE CORRECTIONs # mJy/sr
    peak_flux_star  = nn.Parameter(torch.FloatTensor([photons_normalization / contrast_normalization]))

    offset_STAR = nn.Parameter(torch.FloatTensor([star_offset_x * arcsec2rad(psf_pixel_scale), star_offset_y * arcsec2rad(psf_pixel_scale)]))
    
    # Set up the wavefront objects
    wavefronts_list1 = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_STAR).to(DEVICE) for wl in wlen_weights[0]]

    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = wf_npix
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale_arcsec = psf_pixel_scale
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wlen_weights[0], true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Set up the propagation object
    if fit_second_wfe_offsets is None:
        second_wfe_parametrization = None
    else:
        if fit_second_wfe_offsets == 'grid':
            second_wfe_parametrization = OPDOffsetModule(1024, 1024)
        elif fit_second_wfe_offsets == 'ptt':
            labelled_transmission,segment_centers, segment_labels = load_mirror_segment_info(data_dir=data_dir)
            second_wfe_parametrization = PTT_OPD(labelled_transmission,segment_centers, segment_labels,wf_npix).to(DEVICE)
        elif fit_second_wfe_offsets == 'interpOPD':
            second_wfe_parametrization = LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        elif fit_second_wfe_offsets[:7] == 'zernike':
            numbasis = int(fit_second_wfe_offsets[7:])
            basis = ZernikeBasis(1024, 1024, numbasis)
            second_wfe_parametrization = ZernikeModel(basis)
        elif fit_second_wfe_offsets[:3] == 'IEC':
            numbasis = int(fit_second_wfe_offsets[3:])
            second_wfe_parametrization = IECModesModelOPD(numbasis) #LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        elif fit_second_wfe_offsets[:7] == 'OTEMeas':
            numbasis = int(fit_second_wfe_offsets[7:])
            second_wfe_parametrization = OTEMeasModesModelOPD(numbasis)
        else:
            second_wfe_parametrization = None

    if primaryOPD_basis is None:
        OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)
    else:
        if primaryOPD_basis == 'grid':
            OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)
        elif primaryOPD_basis == 'ptt':
            labelled_transmission,segment_centers, segment_labels = load_mirror_segment_info(data_dir=data_dir)
            OTE_wfe_parametrization = PTT_OPD(labelled_transmission,segment_centers, segment_labels,wf_npix).to(DEVICE)
        elif primaryOPD_basis == 'interpOPD':
            OTE_wfe_parametrization = LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        elif primaryOPD_basis[:7] == 'zernike':
            numbasis = int(primaryOPD_basis[7:])
            basis = ZernikeBasis(1024, 1024, numbasis)
            OTE_wfe_parametrization = ZernikeModel(basis)
        elif primaryOPD_basis[:3] == 'IEC':
            numbasis = int(primaryOPD_basis[3:])
            OTE_wfe_parametrization = IECModesModelOPD(numbasis) #LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        elif primaryOPD_basis[:7] == 'OTEMeas':
            numbasis = int(primaryOPD_basis[7:])
            OTE_wfe_parametrization = OTEMeasModesModelOPD(numbasis)
        else:
            OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)

    prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=oversample,OTE_wfe_basis=OTE_wfe_parametrization,second_delta_wfe=second_wfe_parametrization, num_det_px=num_det_px).to(DEVICE) for _ in range(num_ints)]
    # FIRST OPTIMIZE WITH THE REFERENCE IMAGE
    edge1, edge2 = 160 - int(num_det_px/2), 160 + int(num_det_px/2)

    # NOW WE DO NOT HAVE AN IMAGE CENTERED AT THE ARRAY, so use known coronagraph position to crop at the right pixels
    shift_y,shift_x = -14, 10

    # ref1_data_NOTcentered_CROPPED = ref1_data_NOTcentered[(120-sfhitx):(200-sfhitx),(120-sfhity):(200-sfhity)]

    # We do not use reference files here, so this does not matter
    real_im_ref = np.load(f'{data_dir}/real_data/{reference_file}')[ref_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
    real_im_ref = real_im_ref.astype(np.float32)
    if real_im_ref.ndim == 2:
        real_im_ref = real_im_ref[None]

    reference = real_im_ref
    
    # Only this; formatting must be done prior to calling the function, in contrast with the previous code
    real_im = np.load(measurement_file)[sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
    real_im = real_im.astype(np.float32)
    if real_im.ndim == 2:
        real_im = real_im[None]
    
    observations = real_im


    if injected_companion is not None:
        # inject a planet of a given flux and position to the observation
        with torch.no_grad():
            offset_planet = nn.Parameter(torch.FloatTensor([injected_companion['pos_x_px'] * arcsec2rad(psf_pixel_scale_arcsec), injected_companion['pos_y_px'] * arcsec2rad(psf_pixel_scale_arcsec)]))
            # offset_planet = nn.Parameter(torch.FloatTensor([0.* arcsec2rad(1 / (psf_pixel_scale * 1000)), 0. * arcsec2rad(1 / (psf_pixel_scale * 1000))]))
    
            # Set up the wavefront objects
            wavefronts_list_planet = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_planet).to(DEVICE) for wl in wlen_weights[0]]
            prop_models_planet = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=oversample, num_det_px=num_det_px).to(DEVICE) for _ in range(num_ints)]
            pred_planet = [p_model(wavefronts_list_planet, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models_planet]
            pred_planet = torch.mean(torch.cat(pred_planet, 0), 0)
        
        pred_planet_np = pred_planet.cpu().numpy()
        pred_planet_scaled = pred_planet_np * injected_companion['flux'] /0.0037 / torch.clone(peak_flux_star).detach().numpy() # CHECK THIS AGAIN
        # do strange 0.0037 correction idk
        if ref_cutoff_iter > iters:
            # do everything on reference star
            orig_meas = reference.copy()
            reference+=pred_planet_scaled[None] # inject the planet to the observation
            toplot = reference
        else:
            # actually inject in science target
            orig_meas = observations.copy()
            observations+=pred_planet_scaled[None] # inject the planet to the observation
            toplot = observations


        # real_im+=pred_planet_scaled[None] # inject the planet to the observation
        posxpx_val = injected_companion['pos_x_px']
        posypx_val = injected_companion['pos_y_px']
        radius_inj = injected_companion['radius']
        theta_inj = injected_companion['theta']
        iter_num_inj = injected_companion['iter_num']

        plt.figure(figsize=[15,5])
        plt.subplot(131)
        plt.imshow(orig_meas[0], origin='lower')
        plt.title(f'Sum, peak = {np.nansum(orig_meas):.2f},{np.nanmax(orig_meas):.2f} (orig)')
        plt.subplot(132)
        plt.imshow(pred_planet_scaled, origin='lower')
        plt.title(f'Sum, peak = {np.nansum(pred_planet_scaled):.2f},{np.nanmax(pred_planet_scaled):.2f} (sim planet)')
        plt.subplot(133)
        plt.imshow(toplot[0], origin='lower')
        plt.title(f'Sum, peak = {np.nansum(toplot):.2f},{np.nanmax(toplot):.2f} (sim+orig planet)')
        plt.suptitle(f'X, Y = {posxpx_val}, {posypx_val}')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/vis_measurement_r{radius_inj}_th{theta_inj}_it{iter_num_inj}.png')
        plt.close()
    else:
        pred_planet_scaled = None
    plt.imsave(f'{vis_dir}/vis_measurement_reference.png', reference[0], cmap='viridis', origin='lower')
    # reference = real_im
    reference = torch.from_numpy(reference).to(DEVICE)
    observations = torch.from_numpy(observations).to(DEVICE)

    # print('YES reference median, sum ', reference.detach().median(), reference.detach().sum())
    if not no_median:
        ref_scaled = reference / reference.median()
    else: 
        ref_scaled = reference

    # print('YES ref_scaled median, sum ', ref_scaled.detach().median(), ref_scaled.detach().sum())

    # LOAD REFERENCE IMAGE TOO
    # scale by median before subtraction
    
    
    if not no_median:
        obs_scaled = observations / observations.median()
    else:
        obs_scaled = observations

    # If we want to use noise maps
    if measurement_noisemap is not None:
        real_im_noisemap = np.load(f'{data_dir}/real_data/{measurement_noisemap}')[sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
        real_im_noisemap = real_im_noisemap.astype(np.float32)
        if real_im_noisemap.ndim == 2:
            real_im_noisemap = real_im_noisemap[None]
        
        real_im_noisemap = torch.from_numpy(real_im_noisemap).to(DEVICE)
        
        if not no_median:
            obs_scaled_noisemap = real_im_noisemap / real_im_noisemap.median()
        else:
            obs_scaled_noisemap = real_im_noisemap

    if reference_noisemap is not None:
        reference_im_noisemap = np.load(f'{data_dir}/real_data/{reference_noisemap}')[ref_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
        reference_im_noisemap = reference_im_noisemap.astype(np.float32)
        if reference_im_noisemap.ndim == 2:
            reference_im_noisemap = reference_im_noisemap[None]
        
        reference_im_noisemap = torch.from_numpy(reference_im_noisemap).to(DEVICE)
        
        if not no_median:
            ref_scaled_noisemap = reference_im_noisemap / reference_im_noisemap.median()
        else:
            ref_scaled_noisemap = reference_im_noisemap


    # observations = torch.from_numpy(real_im.astype(np.float32)).to(DEVICE)
    
    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        pred = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred = torch.mean(torch.cat(pred, 0), 0)
    pred_np = pred.cpu().numpy()
    plt.imsave(f'{vis_dir}/vis_PSF_render_init.png', pred_np, cmap='viridis', origin='lower')


    if use_simulated_data:
        with torch.no_grad():
            # Use interpolated WFE as real WFE
            # print('peak flux star is ', peak_flux_star)
            # print('offset star is ', offset_STAR)
            true_offset = torch.clone(offset_STAR)#1.2106e-07, -2.4211e-07]
            true_offset[0]+=2e-7
            true_offset[1]+=1.1e-7

            IEC_RMS1 = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/4050_data_for_diff_modeling/SPIE_modes/IEC_mode_RMS1_1024_meters.npy')
            IEC_RMS1 = np.flip(IEC_RMS1, axis=0)[None]
            IEC_RMS1 = torch.from_numpy(IEC_RMS1.copy()).float()
            IEC_RMS1 = F.interpolate(IEC_RMS1[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
            IEC_RMS1 = IEC_RMS1.contiguous().to(DEVICE)

            wavefronts_list_simulated = [Wavefront(wf_npix, diameter, wl, peak_flux_star*100, true_offset).to(DEVICE) for wl in wlen_weights[0]]
            wfe_sim_interp = wfe_batch_list[0] + (wfe_batch_list[1] - wfe_batch_list[0]) * 0.5
            wfe_sim_interp_series = [wfe_sim_interp + (1*IEC_RMS1), wfe_sim_interp + (2*IEC_RMS1)]
            sim = [prop_models[j](wavefronts_list_simulated, wfe_sim_interp_series[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
            # print('peak of sim pre mean is ', torch.max(torch.cat(sim)))
            sim = torch.mean(torch.cat(sim, 0), 0)
            # print('shape of sim is ', sim)
            # print('peak is ', torch.max(sim))
            # print('sqrt peak is ', torch.sqrt(torch.max(sim)))
            shot_noise = torch.normal(mean = sim, std = 1. * torch.sqrt(sim)) - sim
            read_noise = torch.normal(mean = 0, std = 42. * torch.ones_like(sim))
            noise = shot_noise + read_noise

            out_noisy = sim + noise
            sim = torch.clamp(out_noisy, min=0.0)

            np.save(f'{vis_dir}/simulated_obs_no_planet.npy',sim.detach().cpu().numpy())
            # Whatever, set observation and reference as the same
            observations = torch.clone(sim[None])
            observations = shift_image_subpixel(observations)
            if not no_median:
                obs_scaled = observations / observations.median()
            else:
                obs_scaled = observations

            reference = torch.clone(sim[None]) 
            reference = shift_image_subpixel(reference)
            if not no_median:
                ref_scaled = reference / reference.median()
            else:
                ref_scaled = reference

    if px_mask_file is not None:
        px_mask = np.load(f'{data_dir}/real_data/{px_mask_file}')
        if num_det_px != 80:
            edge1, edge2 = 40 - int(num_det_px/2), 40 + int(num_det_px/2)
            px_mask = px_mask[edge1:edge2, edge1:edge2]
        px_mask = torch.from_numpy(px_mask.astype(np.float32)).to(DEVICE)[None]

    if not no_median:
        pred_scaled = pred / pred.median()
    else:
        pred_scaled = pred 

    
    est_residual = (obs_scaled - pred_scaled).detach().cpu().mean(0).numpy()
    print('est residual shape is ', est_residual.shape)
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')

    est_ref_residual = (obs_scaled - ref_scaled).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_ref_res_init.png', est_ref_residual, cmap='viridis', origin='lower')

    # scale by median before subtraction
    if not no_median:
        obs_scaled = observations / observations.median()
        pred_scaled = pred / pred.median()
    else:
        obs_scaled = observations
        pred_scaled = pred
    est_residual = (obs_scaled - pred_scaled).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')

    # Set up the optimizer and scheduler
    
    """
    wfe_offsets: learns to offset the wfe_batch
    fpm_shifts: learns to shift the focal plane mask
    angle_offsets: learns to offset the star incident angle
    lyot_shifts: learns to shift the lyot mask
    nircam_offsets: learns to offset the nircam opd
    """
    
    if be_normal:
        optics_params = list()
        for p_model in prop_models:
            optics_params+=list(p_model.angle_offsets.parameters())
            if fit_Lyot:
                optics_params+=list(p_model.lyot_shifts.parameters())

            optics_params +=  list(p_model.nircam_offsets.parameters()) + list(p_model.wfe_offsets.parameters()) #+ list(p_model.charge_diffusion.parameters()) #+ list(p_model.flux_correction.parameters())
            if fit_everything:
                optics_params+= list(p_model.fpm_shifts.parameters()) + list(p_model.lyot_shifts.parameters()) + list(p_model.flux_correction.parameters())

        # print('PARAMS ARE ', optics_params)
        # optimizer = torch.optim.AdamW(optics_params, lr=args.lr, weight_decay=0.0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=args.lr)

        optimizer_sgd = torch.optim.SGD(optics_params, lr=lr, momentum=0.9)
        optimizer_adam = torch.optim.AdamW(optics_params, lr=lr, weight_decay=0.0)

        scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=iters, eta_min=lr)
        scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=iters, eta_min=lr)
    
    else:
        optics_params_normal_lr = list()
        optics_params_bigger_lr = list()

        bigger_lr = lr*big_lr_factor

        for p_model in prop_models:
            if fit_Lyot:
                optics_params_bigger_lr+=list(p_model.lyot_shifts.parameters())
            optics_params_bigger_lr+=list(p_model.angle_offsets.parameters())
            optics_params_normal_lr+=list(p_model.nircam_offsets.parameters()) + list(p_model.wfe_offsets.parameters()) #+ list(p_model.charge_diffusion.parameters()) #+ list(p_model.flux_correction.parameters())

        optimizer_argument = [
            {'params': optics_params_normal_lr, 'lr': lr},
            {'params': optics_params_bigger_lr, 'lr': bigger_lr}
        ]

        optimizer_sgd = torch.optim.SGD(optimizer_argument, momentum=0.9)
        optimizer_adam = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

        scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=iters, eta_min=lr)
        scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=iters, eta_min=lr)

    progress_arr_reference = []
    opd_vis_arr = []
    opd_vis_offset_arr = []

    if fit_second_wfe_offsets is not None:
        second_opd_vis_offset_arr = []

    if num_ints > 1:
        opd_vis_arr1 = []
        opd_vis_offset_arr_1 = []
        # opd_vis_arr2 = []
    nircam_opd_vis_arr = []
    nircam_opd_vis_arr_1 = []

    angles_offset_res = []
    angles_offset_res_1 = []

    residual_max_arr = []

    progress_arr_target = []

    if fit_Lyot:
        lyot_arr = []

    progress_fluxcorrection = []

    cutoff_iter = ref_cutoff_iter

    switch_step = 0

    # PRIORS!
    # position_center_prior = torch.tensor([0., 0.], device=DEVICE)
    # sigma_center_prior = 1e-5
    if pre_initialized_state is None:
        tbar = tqdm.tqdm(range(iters + 1))
    else:
        # Start iterations from science target, taking as initial conditions the pre initialized state (obtained through the first pass, so we don't need to pre train every time)
        tbar = tqdm.tqdm(range(cutoff_iter,iters + 1)) # start iterating at cutoff_iter+1, so start directly on science target
        for j in range(len(prop_models)):
            prop_models[j].load_state_dict(copy.deepcopy(pre_initialized_state['pointpropagate_list_state_dict'][j]))
        optimizer_adam.load_state_dict(copy.deepcopy(pre_initialized_state['optimizer_state_dict']))
        scheduler_adam.load_state_dict(copy.deepcopy(pre_initialized_state['scheduler_state_dict']))

        tbar = tqdm.tqdm(range(pre_initialized_state['scheduler_state_dict']['last_epoch'] + 1,iters + 1)) # start iterating at cutoff_iter+1, so start directly on science target
        print('PRE INITIALIZED STATE LOADED IS ', pre_initialized_state)
    
    # Start optimization
    for i in tbar:
        optimizer = optimizer_sgd if i < switch_step else optimizer_adam
        scheduler = scheduler_sgd if i < switch_step else scheduler_adam
        if pre_initialized_state is not None:
            print('AT ITERATION ', i, ' WE HAVE THIS OBJECT STATE ',prop_models[0].state_dict())
            print('AT ITERATION ', i, ' WE HAVE THIS optim STATE ',optimizer.state_dict())
            print('AT ITERATION ', i, ' WE HAVE THIS sched STATE ',scheduler.state_dict())
        
        optimizer.zero_grad()
        if i == iters:
            pred_1 = []
            print('ENTERING PHASOR')
            for j in range(len(prop_models)):
                res, wf = prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0],save_wf=True)
                print('RESULT PHASOR')
                pred_1.append(res)
                wfnumpy = wf[0].detach().cpu().numpy()
                print('DETACHED PHASOR')
                np.save(f'{vis_dir}/last_iteration_TOTALWAVEFRONT_oversample_{oversample}_wl_sampling_{num_wl}.npy',wfnumpy)
                print('SAVED PHASOR')
        else:
            pred_1 = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]

        pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]

        # if i%100==0:
        #     print('YES pred_1 median, sum ', pred_1.detach().median(), pred_1.detach().sum())

        # compute loss in median-scaled space
        if not no_median:
            pred_scaled = pred_1 / pred_1.detach().median()
        else:
            pred_scaled = pred_1
        # if i%100==0:
        #     print('YES pred_scaled median, sum ', pred_scaled.detach().median(), pred_scaled.detach().sum())
        est_residual_obs = obs_scaled - pred_scaled.detach()

        est_residual_ref = ref_scaled - pred_scaled.detach()
        # if i%100==0:
        #     with torch.no_grad():
                # print('YES flux factor correction 1 is ', prop_models[0].flux_correction.forward(1.))
                # # print('YES flux factor correction 2 is ', prop_models[1].flux_correction.forward(0.))
                # print('YES flux angles 1 is ', prop_models[0].angle_offsets.forward())
                # # print('YES flux angles 2 is ', prop_models[1].angle_offsets.forward())
                # print('pred scaled max is ', pred_scaled.max())
                # print('original data scaled is ', ref_scaled.max())

        if px_mask_file is not None:
            if reference_noisemap is not None:
                global_l1_loss = weighted_smooth_l1_loss(pred_scaled, ref_scaled, ref_scaled_noisemap,reduction='none')
            else:
                global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled, reduction='none')
            
            if measurement_noisemap is not None:
                obs_l1_loss = weighted_l1_loss(pred_scaled, obs_scaled, obs_scaled_noisemap,reduction='none')
            else:
                obs_l1_loss = F.l1_loss(pred_scaled, obs_scaled, reduction='none')

            global_l1_loss = global_l1_loss * px_mask
            global_l1_loss = global_l1_loss.sum() / px_mask.sum()  

            obs_l1_loss = obs_l1_loss * px_mask
            obs_l1_loss = obs_l1_loss.sum() / px_mask.sum()  
        else:
            # global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled)
            # obs_l1_loss = F.l1_loss(pred_scaled, obs_scaled)
            if reference_noisemap is not None:
                global_l1_loss = weighted_smooth_l1_loss(pred_scaled, ref_scaled, ref_scaled_noisemap)
            else:
                global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled)
            
            if measurement_noisemap is not None:
                obs_l1_loss = weighted_l1_loss(pred_scaled, obs_scaled, obs_scaled_noisemap)
            else:
                obs_l1_loss = F.l1_loss(pred_scaled, obs_scaled)

        if i > cutoff_iter:
            if i < (stage_fluxpos_cutoff_iter + cutoff_iter):
                loss = obs_l1_loss
            else:
                loss = sci_lr_weight*obs_l1_loss
        else:
            loss = global_l1_loss

        if smooth is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            if num_ints > 1:
                opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])
            
            TVL = total_variation_loss(opd1)
            if num_ints > 1:
                TVL+= total_variation_loss(opd2)
                

            nircam_opd = prop_models[0].nircam_offsets.get_res()
            if num_ints > 1:
                nircam_op1 = prop_models[1].nircam_offsets.get_res()
            
            TVL += total_variation_loss(nircam_opd)
            if num_ints > 1:
                TVL += total_variation_loss(nircam_op1)

            if i > cutoff_iter:
                # loss = 0.001*obs_l1_loss
                loss = loss + smooth * TVL
            else:
                loss = loss + smooth * TVL



        if num_ints > 1 and OPD_loss_weight is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])

            opd1_scaled = opd1 
            opd2_scaled = opd2 

            opds_disimilarity = F.smooth_l1_loss(opd1_scaled, opd2_scaled)

            if i > cutoff_iter:
                # loss = 0.001*obs_l1_loss
                loss = loss + OPD_loss_weight * opds_disimilarity * 1e15
            else:
                loss = loss + OPD_loss_weight * opds_disimilarity * 1e15

        elif args.OPD_loss_weight is not None:
            opd = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])

            opds_disimilarity_before = F.smooth_l1_loss(opd, wfe_batch_list[0])
            opds_disimilarity_after = F.smooth_l1_loss(opd, wfe_batch_list[1])

            opds_total_disimilarity = opds_disimilarity_before + opds_disimilarity_after

            if i > cutoff_iter:
                loss = loss + args.OPD_loss_weight * opds_total_disimilarity * 1e15
            else:
                loss = loss + args.OPD_loss_weight * opds_total_disimilarity * 1e15


        loss.backward()
        
        if fit_second_wfe_offsets is not None:
            if i > cutoff_iter:
                # Do not update original WFE
                for p_model in prop_models:
                    for t in p_model.wfe_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0
                    for t in p_model.nircam_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0
                if i < (stage_fluxpos_cutoff_iter + cutoff_iter):
                    for p_model in prop_models:
                        for t in p_model.second_wfe_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0
            else:
                # Do not update second delta wfe
                for p_model in prop_models:
                    for t in p_model.second_wfe_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0

        if i > cutoff_iter:
            if freeze_OTE_OPD_sci:
                for p_model in prop_models:
                    for t in p_model.wfe_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0
            if i < (stage_fluxpos_cutoff_iter + cutoff_iter):
                with torch.no_grad():
                    fluxwindsize = 80
                    maskslice = px_mask[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]
                    tot_flux_pred = (maskslice*pred_scaled[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                    tot_flux_target = (maskslice*obs_scaled[:,(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                    flux_mismatch_ratio = tot_flux_target / tot_flux_pred
                    if i%100 == 0:
                        print('flux mismatch ratio SCI is ', flux_mismatch_ratio)
                    for p_model in prop_models:
                        p_model.flux_correction.data *=flux_mismatch_ratio.to(DEVICE)#nn.Parameter(flux_mismatch_ratio.to(DEVICE))
                    for p_model in prop_models:
                        for t in p_model.wfe_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0
                        for t in p_model.nircam_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0

            else:
                if freeze_position:
                    for p_model in prop_models:
                        for t in p_model.angle_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0
        elif i < stage_fluxpos_cutoff_iter:
            # Try manual fit of the flux, since optimization is being funny
            with torch.no_grad():
                fluxwindsize = 80
                maskslice = px_mask[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]
                tot_flux_pred = (maskslice*pred_scaled[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                if i > cutoff_iter:
                    tot_flux_target = (maskslice*obs_scaled[:,(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                else:
                    tot_flux_target = (maskslice*ref_scaled[:,(int(ref_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(ref_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(ref_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(ref_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()

                flux_mismatch_ratio = tot_flux_target / tot_flux_pred
                if i%100 == 0:
                    print('flux mismatch ratio is ', flux_mismatch_ratio)
                for p_model in prop_models:
                    p_model.flux_correction.data *=flux_mismatch_ratio.to(DEVICE)#nn.Parameter(flux_mismatch_ratio.to(DEVICE))
                
            for p_model in prop_models:
                for t in p_model.wfe_offsets.parameters():
                    if t.requires_grad:
                        t.grad *= 0.0
                for t in p_model.nircam_offsets.parameters():
                    if t.requires_grad:
                        t.grad *= 0.0
        else:
            if freeze_position:
                for p_model in prop_models:
                    for t in p_model.angle_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0

        optimizer.step()
        scheduler.step()

        # if i == cutoff_iter and pre_initialized_state is None:
        #     # Save everything! This is the last iteration, so we will use it to iterate again in the future
        #     # Save state of the model parameters, state of the optimizer and state of the scheduler 
        #     final_ref_state = {
        #         'pointpropagate_list_state_dict': [copy.deepcopy(p_model.state_dict()) for p_model in prop_models],
        #         'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        #         'scheduler_state_dict': copy.deepcopy(scheduler.state_dict())
        #     }

        #     print('FINAL STATE WHEN SAVED ', final_ref_state)

        if i > cutoff_iter:
            progress_arr_target.append(est_residual_obs[0])
        else:
            progress_arr_target.append(est_residual_obs[0])
            progress_arr_reference.append(est_residual_ref[0])

        
        if i % vis_freq == 0 and i > cutoff_iter:
            plt.imsave(f'{vis_dir}/vis_est_res_TARG_{i}.png', progress_arr_target[-1].cpu().numpy(), cmap='viridis', origin='lower')
        elif i % vis_freq == 0:
            plt.imsave(f'{vis_dir}/vis_est_res_REF_{i}.png', progress_arr_reference[-1].cpu().numpy(), cmap='viridis', origin='lower')
        
        cur_opd = prop_models[0].wfe_offsets.forward(wfe_batch_list[0]).squeeze().detach().cpu()
        opd_vis_arr.append(cur_opd)

        if fit_Lyot:
            with torch.no_grad():
                result= prop_models[0].lyot_shifts(prop_models[0].lyot)
                # print('RESULT shape IS ', result.shape)
                lyot_arr.append(result.detach().cpu())

        opd_vis_offset_arr.append(prop_models[0].wfe_offsets.get_res().squeeze().detach().cpu())

        if fit_second_wfe_offsets is not None:
            second_opd_vis_offset_arr.append(prop_models[0].second_wfe_offsets.get_res().squeeze().detach().cpu())

        angles_offset_res.append(prop_models[0].angle_offsets().squeeze().detach().cpu())
        progress_fluxcorrection.append(prop_models[0].flux_correction(1.))
        if num_ints > 1:
            cur_opd = prop_models[1].wfe_offsets.forward(wfe_batch_list[1]).squeeze().detach().cpu()
            opd_vis_arr1.append(cur_opd)

            angles_offset_res_1.append(prop_models[1].angle_offsets())

            opd_vis_offset_arr_1.append(prop_models[1].wfe_offsets.get_res().squeeze().detach().cpu())

            curr_nircam_opd_1 = prop_models[1].nircam_offsets.get_res().squeeze().detach().cpu()
            nircam_opd_vis_arr_1.append(curr_nircam_opd_1)

        curr_nircam_opd = prop_models[0].nircam_offsets.get_res().squeeze().detach().cpu()
        nircam_opd_vis_arr.append(curr_nircam_opd)

        
        tbar_out = {'loss': global_l1_loss.item()}
        tbar.set_postfix(tbar_out)

    # if pre_initialized_state is None:
    #     progress_arr = torch.stack(progress_arr_reference).cpu().numpy()[::10]
    #     progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    #     progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    #     progress_arr = np.flip(progress_arr, 1)
    #     imageio.mimsave(f'{vis_dir}/progress_REFERENCE.mp4', progress_arr, 
    #                     'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })
    
    psf_pixel_scale = 0.062424185
    target_numpys = np.squeeze(torch.stack(progress_arr_target).cpu().numpy())
    opd_vis_arr_numpys = np.squeeze(opd_vis_arr[-1].cpu().numpy())

    if fit_Lyot:
        lyot_arr_numpys = np.squeeze(lyot_arr[-1].cpu().numpy())

    opd_vis_offset_arr_numpys = np.squeeze(opd_vis_offset_arr[-1].cpu().numpy())

    if fit_second_wfe_offsets is not None:
        second_opd_vis_offset_arr_numpys = np.squeeze(second_opd_vis_offset_arr[-1].cpu().numpy())

    if pre_initialized_state is None:
        reference_numpys = np.squeeze(torch.stack(progress_arr_reference).cpu().numpy())

    angles_offset_res_numpys = np.squeeze(angles_offset_res[-1].detach().cpu().numpy())
    angles_offset_res_all = np.squeeze(torch.stack(angles_offset_res).cpu().numpy())

    np.save(f'{vis_dir}/last_iteration_ALL_ANGLES_OFFSETS_rad_oversample_{oversample}_wl_sampling_{num_wl}.npy',angles_offset_res_all)

    # plot converting to mas
    plt.figure(figsize=[10,5])
    plt.subplot(121)
    plt.plot(angles_offset_res_all[:,0] * 206264806.2471)
    plt.grid()
    plt.title('x (?) position offset (mas)')

    plt.subplot(122)
    plt.plot(angles_offset_res_all[:,1] * 206264806.2471)
    plt.grid()
    plt.title('y (?) position offset (mas)')

    plt.tight_layout()
    plt.savefig(f'{vis_dir}/last_iteration_ALL_ANGLES_OFFSETS_HISTORYPLOT_mas_oversample_{oversample}_wl_sampling_{num_wl}.png')
    plt.close()

    progress_fluxcorrection_numpys = np.squeeze(progress_fluxcorrection[-1].detach().cpu().numpy())

    if num_ints > 1:
        opd_vis_arr_numpys1 = np.squeeze(opd_vis_arr1[-1].cpu().numpy())
        angles_offset_res_numpys_1 = np.squeeze(angles_offset_res_1[-1].detach().cpu().numpy())
        opd_vis_offset_arr_numpys_1 = np.squeeze(opd_vis_offset_arr_1[-1].cpu().numpy())

    nircam_opd_vis_arr_numpys = np.squeeze(nircam_opd_vis_arr[-1].cpu().numpy())
    if num_ints > 1:
        nircam_opd_vis_arr_1_numpys = np.squeeze(nircam_opd_vis_arr_1[-1].cpu().numpy())

    opd_vis_arr_numpys_initial = np.squeeze(opd_vis_arr[0].cpu().numpy())
    if num_ints > 1:
        opd_vis_arr_numpys1_initial = np.squeeze(opd_vis_arr1[0].cpu().numpy())
        # opd_vis_arr_numpys2 = np.squeeze(opd_vis_arr2[-1].cpu().numpy())
    nircam_opd_vis_arr_numpys_initial = np.squeeze(nircam_opd_vis_arr[0].cpu().numpy())

    if injected_companion is not None:
        folder_iteration_dir = os.path.join(vis_dir,f'run_r{radius_inj}_theta{theta_inj}')
        if not os.path.isdir(folder_iteration_dir):
            os.mkdir(folder_iteration_dir)
    else:
        folder_iteration_dir = vis_dir

    np.save(f'{folder_iteration_dir}/last_iteration_ENTRANCE_OPD_oversample_{oversample}_wl_sampling_{num_wl}.npy',opd_vis_arr_numpys)
    np.save(f'{folder_iteration_dir}/last_iteration_ENTRANCE_OPD_OFFSET_oversample_{oversample}_wl_sampling_{num_wl}.npy',opd_vis_offset_arr_numpys)

    if fit_second_wfe_offsets is not None:
        np.save(f'{vis_dir}/last_iteration_SECONDDELTA_ENTRANCE_OPD_OFFSET_oversample_{oversample}_wl_sampling_{num_wl}.npy',second_opd_vis_offset_arr_numpys)

    np.save(f'{folder_iteration_dir}/last_iteration_ANGLES_OFFSET_oversample_{oversample}_wl_sampling_{num_wl}.npy',angles_offset_res_numpys)
    np.save(f'{folder_iteration_dir}/last_iteration_FLUXCORRECTION_oversample_{oversample}_wl_sampling_{num_wl}.npy',progress_fluxcorrection_numpys)
    

    if num_ints > 1:
        np.save(f'{folder_iteration_dir}/last_iteration_ENTRANCE_OPD_oversample_{oversample}_wl_sampling_{num_wl}_1.npy',opd_vis_arr_numpys1)
        np.save(f'{folder_iteration_dir}/last_iteration_ANGLES_OFFSET_oversample_{oversample}_wl_sampling_{num_wl}_1.npy',angles_offset_res_numpys_1)
        np.save(f'{folder_iteration_dir}/last_iteration_ENTRANCE_OPD_OFFSET_oversample_{oversample}_wl_sampling_{num_wl}_1.npy',opd_vis_offset_arr_numpys_1)

    np.save(f'{folder_iteration_dir}/last_iteration_NIRCAM_OPD_oversample_{oversample}_wl_sampling_{num_wl}.npy',nircam_opd_vis_arr_numpys)
    if num_ints > 1:
        np.save(f'{folder_iteration_dir}/last_iteration_NIRCAM_OPD_oversample_{oversample}_wl_sampling_{num_wl}_1.npy',nircam_opd_vis_arr_1_numpys)
        

    np.save(f'{folder_iteration_dir}/first_iteration_ENTRANCE_OPD_oversample_{oversample}_wl_sampling_{num_wl}.npy',opd_vis_arr_numpys_initial)
    if num_ints > 1:
        np.save(f'{folder_iteration_dir}/first_iteration_ENTRANCE_OPD_oversample_{oversample}_wl_sampling_{num_wl}_1.npy',opd_vis_arr_numpys1_initial)
        # np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_2.npy',opd_vis_arr_numpys2)
    np.save(f'{folder_iteration_dir}/first_iteration_NIRCAM_OPD_oversample_{oversample}_wl_sampling_{num_wl}.npy',nircam_opd_vis_arr_numpys_initial)


    x_pos_real, y_pos_real = -7.2 * psf_pixel_scale, 11 * psf_pixel_scale # hand tuned for HIP 65426 in the pre-rotated frame

    np.save(f'{folder_iteration_dir}/last_iteration_oversample_{oversample}_wl_sampling_{num_wl}.npy', target_numpys[-1])
    if pre_initialized_state is None:
        np.save(f'{folder_iteration_dir}/last_iteration_REFERENCE_oversample_{oversample}_wl_sampling_{num_wl}.npy', reference_numpys[-1])
    
    np.save(f'{folder_iteration_dir}/last_iteration_MODEL_oversample_{oversample}_wl_sampling_{num_wl}.npy', pred_scaled.detach().cpu().numpy())

    progress_arr = torch.stack(progress_arr_target).cpu().numpy()[::100]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress_TARGET.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })


    opd_vis_arr = torch.stack(opd_vis_arr)[::100]
    opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
    opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
    opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
    # print('opd vis arr shape is ', opd_vis_arr.shape)
    imageio.mimsave(f'{vis_dir}/opd_progress.mp4', opd_vis_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
    
    if fit_Lyot:
        lyot_arr = torch.stack(lyot_arr)[::10]
        lyot_arr = (lyot_arr - lyot_arr.min()) / (lyot_arr.max() - lyot_arr.min())
        lyot_arr = np.uint8(cm.binary(lyot_arr.numpy())* 255)
        lyot_arr = np.squeeze(lyot_arr)
        imageio.mimsave(f'{vis_dir}/lyot_progress.mp4', lyot_arr, 
                        'FFMPEG', **{'macro_block_size': None, 'fps': 30, })

    if pre_initialized_state is None:
        if ref_cutoff_iter > iters:
            return reference_numpys[-1], np.nanmax(pred_planet_scaled)#, final_ref_state
        else:
            return progress_arr_target[-1].cpu().numpy(), np.nanmax(pred_planet_scaled)#,final_ref_state
    else:
        # Return the usual stuff
        if ref_cutoff_iter > iters:
            return reference_numpys[-1], np.nanmax(pred_planet_scaled)
        else:
            return progress_arr_target[-1].cpu().numpy(), np.nanmax(pred_planet_scaled)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--measurement_file', default='justdata_bothintegrations.npy', type=str)
    parser.add_argument('--reference_file', default='reference_00001.npy', type=str)
    parser.add_argument('--scene_name', default='HIP65426', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--lr', default=1e-8, type=float)
    parser.add_argument('--star_offset_x', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--star_offset_y', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--use_ptt', action='store_true')
    parser.add_argument('--use_linear_interp_opd', action='store_true')
    parser.add_argument('--num_wl', default=None, type=int)
    parser.add_argument('--oversample', default=None, type=int)
    parser.add_argument('--fit_everything', action='store_true')
    parser.add_argument('--num_det_px', default=80, type=int)
    parser.add_argument('--num_ints', default=1, type=int)
    parser.add_argument('--OPD_loss_weight', default=None, type=float)
    parser.add_argument('--smooth', default=None, type=float)
    parser.add_argument('--use_simulated_data', action='store_true')
    parser.add_argument('--no_median', action='store_true')
    parser.add_argument('--stage_fluxpos_cutoff_iter', default=0, type=int)
    parser.add_argument('--sci_targ_name', default=None, type=str)
    parser.add_argument('--ref_cutoff_iter', default=500, type=int)
    parser.add_argument('--px_mask_file', default=None, type=str)
    parser.add_argument('--ref_which_int', default=0, type=int)
    parser.add_argument('--sci_which_int', default=0, type=int)
    parser.add_argument('--reference_noisemap', default=None, type=str)
    parser.add_argument('--measurement_noisemap', default=None, type=str)
    parser.add_argument('--fit_Lyot', action='store_true')
    parser.add_argument('--freeze_position', action='store_true')
    parser.add_argument('--be_normal', action='store_true')
    parser.add_argument('--sci_lr_weight', default=1., type=float)
    parser.add_argument('--fit_second_wfe_offsets', default=None, type=str)
    parser.add_argument('--primaryOPD_basis', default=None, type=str)
    parser.add_argument('--other_OPD_meas', default=None, type=str)
    parser.add_argument('--big_lr_factor', default=100., type=float)
    parser.add_argument('--freeze_OTE_OPD_sci', action='store_true')
    parser.add_argument('--insert_initial_delta_OTE_OPD', action='store_true')
    parser.add_argument('--insert_initial_delta_NIRCam_OPD', action='store_true')
    parser.add_argument('--which_survey', default=None, type=str)

    args = parser.parse_args()

    DEVICE = 'cuda'

    this_vis_freq = 300

    if args.which_survey == '4050':
        superfolder = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/gooddata4050_5835/prepared_4050_obs'
        obs_info = pd.read_csv('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/gooddata4050_5835/prepared_4050_obs/obs_info_4050_for_autodiff.csv')
    elif args.which_survey == '5835':
        superfolder = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/gooddata4050_5835/prepared_5835_obs_PREMETEOR'
        obs_info = pd.read_csv('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/gooddata4050_5835/prepared_5835_obs_PREMETEOR/obs_info_5835_for_autodiff.csv')

    root_dir = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/vis/systematic_'+args.which_survey+'_no_smoothing'

    for targprop in list(obs_info['TARGPROP'].unique()):
        # Get all observations from a given target. Usually, science targets have one observation, and references have 9 from the 9 dithers
        observations = sorted(glob.glob(superfolder+'/'+targprop+'/full_im/*'))

        # Loop through all observations
        for curr_obs in observations:
            # date_obs is a date in format like 'YYYY-MM-DD'
            # Now, we have to find 
            date_obs, obs_number, dither_number = get_date_from_obs(curr_obs, obs_info,args.which_survey)
            before_and_after_OPD = get_before_and_after_opd(date_obs)
            correct_OPD_array_as_loaded = before_and_after_OPD.copy()

            scene_name = targprop+'/'+obs_number+'/'+dither_number

            # Load observation to extract number of integrations
            obs = np.load(curr_obs)
            num_ints = len(obs)

            for curr_int in range(num_ints):
                # It is at this level that we run our optimization
                # From the input parameters, only modify:
                # - Integration number: change for curr_int
                # - Measurement file: change for curr_obs
                # - correct_OPD_array_as_loaded is added: just identify and load the correct OPD given the date of the observation: flip is done
                #      inside run_optimization, so don't do it again up here
                # - root_dir is the big folder: whether 4050 or 5835
                # - scene_name has format targprop/obs_num/dith_num
                # - exp_name is the integration number
                # - vis_freq is also changed to save space
                # Also, given the volume of data, do not save videos as usual: save only every 100th frame (yeah) as opposed to very 5th
                exp_name = 'integration_'+str(curr_int)
                residual, _  = run_optimization(args.data_dir, root_dir, scene_name,exp_name, args.star_offset_x, args.star_offset_y, args.use_ptt,args.reference_file, curr_obs,args.lr,args.iters, this_vis_freq,args.ref_cutoff_iter, args.num_wl, args.oversample,args.num_det_px,args.sci_targ_name,args.num_ints,args.ref_which_int, curr_int,args.use_linear_interp_opd,args.no_median,args.measurement_noisemap,args.reference_noisemap,args.use_simulated_data,args.px_mask_file,args.fit_Lyot,args.fit_everything,args.stage_fluxpos_cutoff_iter,args.smooth,args.OPD_loss_weight,args.freeze_position,args.be_normal,args.sci_lr_weight,args.fit_second_wfe_offsets, args.primaryOPD_basis,args.other_OPD_meas,args.big_lr_factor,args.freeze_OTE_OPD_sci,args.insert_initial_delta_OTE_OPD,args.insert_initial_delta_NIRCam_OPD, correct_OPD_array_as_loaded=correct_OPD_array_as_loaded)
                print('Done with '+args.which_survey + ' program with target '+ targprop + ' and integration ' + str(curr_int) + '/'+str(num_ints)+' !!')

