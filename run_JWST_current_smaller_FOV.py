import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

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
        self.rotation = nn.Parameter(torch.zeros(1))
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


class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, args,oversample=None, use_ptt = None, num_det_px = 80):
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
        self.fpm_shifts = ShiftModule(fpm.shape[-2], fpm.shape[-1])
        self.nircam_offsets = GridOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])

        self.flux_correction = FluxOffsetModule()

        if use_ptt is None:
            self.wfe_offsets = OPDOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])
        else:
            self.wfe_offsets = use_ptt
            
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


    def forward(self, wavefront_list, wfe, wl_weights, wavelenghts):
        output = None
        wfe_ = self.wfe_offsets(wfe)
        for i in range(len(wl_weights)):
            wavefront = wavefront_list[i]
            phasor = wavefront.get_phasor(self.angle_offsets())
            phasor = wavefront.forward_wfe(phasor, wfe_, wavelenghts[i])
            phasor_ap = wavefront.forward(phasor, self.aperture, normalize=True)
            phasor_fpm = wavefront.forward_fpm(phasor_ap, self.fpm_shifts(self.fpm[:, i]),oversample=self.oversample)
            phasor_lyot = wavefront.forward(phasor_fpm, self.lyot_shifts(self.lyot))
            phasor_nircam_opd = wavefront.forward_wfe(phasor_lyot, self.nircam_offsets(self.nircam_opd[:, i]), wavelenghts[i])
            phasor = (self.y_mat[i].T @ phasor_nircam_opd) @ self.x_mat[i]
            phasor *= self.mult[i]
            # w = (wavefront.peak_flux) ** 0.5
            w = (wavefront.peak_flux) ** 0.5
            out = (torch.abs(phasor) * w) ** 2 
            # out = (torch.abs(phasor) * self.flux_correction(0.)**0.5) ** 2 
            if self.oversample != 1:
                # Rebin while conserving flux
                out = torch.sum(torch.reshape(out, (1,self.num_det_px,self.oversample,self.num_det_px,self.oversample)), (2,4))

            if output is None:
                output = out * wl_weights[i]
            else:
                output += out * wl_weights[i]

        # output = torch.flip(output, dims=(-2,))

        # output = torch.mul(output,self.flux_correction())

        output = self.flux_correction(output)

        # Valid for NIRCam; right sigma probably depends on filter/detector/etc
        # Charge diffusion, probably the most significant detector effect at play here
        # Other detector effects can be added as convolutions with the kernels in the data/detector_kernels folder
        output = v2.GaussianBlur(kernel_size=3, sigma=0.28)(output)

        return output

    def forward_val(self, wavefront):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward(phasor, self.aperture)
        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.peak_flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--measurement_file', default='justdata_bothintegrations.npy', type=str)
    parser.add_argument('--reference_file', default='reference_00001.npy', type=str)
    parser.add_argument('--scene_name', default='HIP65426', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--vis_freq', default=50, type=int)
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
    

    args = parser.parse_args()

    DEVICE = 'cuda'
    # Number of pixels in the wavefront
    wf_npix = 1024
    # Diameter of the aperture
    diameter = 6.603464
    # Number of pixels in the PSF
    psf_npix = args.num_det_px
    # Pixel scale in our detector, in arcseconds. For our case, around 14 miliarcseconds per pixel is expected.
    psf_pixel_scale = 0.062424185

    # Load data from args.data_dir
    if args.oversample is None and args.num_wl is None:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=args.data_dir)
    else:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=args.data_dir, num_wl=args.num_wl, oversample=args.oversample)
        
     
    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]
    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights)

    sampledWFEs = np.load(f'{args.data_dir}/masks_2048/observation_opd.npy')
    sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch = sampledWFEs.contiguous().to(DEVICE)

    if args.sci_targ_name is None:
        sampledWFEs_after = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/4050_data_for_diff_modeling/group_2_after/masks_2048/observation_opd.npy')
    elif args.sci_targ_name == 'HR8799':
        sampledWFEs_after = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HR8799stuff/after_data/masks_2048/observation_opd.npy')
    sampledWFEs_after = np.flip(sampledWFEs_after, axis=0)[None]
    sampledWFEs_after = torch.from_numpy(sampledWFEs_after.copy()).float()
    sampledWFEs_after = F.interpolate(sampledWFEs_after[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch_after = sampledWFEs_after.contiguous().to(DEVICE)


    wfe_batch_list = [wfe_batch, wfe_batch_after]
    ############
    # Set up export directories
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}/{args.exp_name}/initial_fit_reference'
    os.makedirs(vis_dir, exist_ok=True)

    contrast_normalization = 0.003716380479003919
    sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data; VERY ROUGH! To be fixed
    photons_normalization = 90578.00102527262 * sim_to_real_scaling *1e4 / 2251.24456327477#THREE CORRECTIONs # mJy/sr
    peak_flux_star  = nn.Parameter(torch.FloatTensor([photons_normalization / contrast_normalization]))

    offset_STAR = nn.Parameter(torch.FloatTensor([args.star_offset_x * arcsec2rad(psf_pixel_scale), args.star_offset_y * arcsec2rad(psf_pixel_scale)]))
    
    # Set up the wavefront objects
    wavefronts_list1 = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_STAR).to(DEVICE) for wl in wlen_weights[0]]

    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = wf_npix
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wlen_weights[0], true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Set up the propagation object
    if args.use_ptt:
        labelled_transmission,segment_centers, segment_labels = load_mirror_segment_info(data_dir=args.data_dir)
        entrance_OPD_PTT = PTT_OPD(labelled_transmission,segment_centers, segment_labels,wf_npix).to(DEVICE)
        prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=args.oversample, use_ptt=entrance_OPD_PTT, num_det_px=args.num_det_px).to(DEVICE) for _ in range(args.num_ints)]
    elif args.use_linear_interp_opd:
        entrance_OPD_LinearInterp = LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=args.oversample, use_ptt=entrance_OPD_LinearInterp, num_det_px=args.num_det_px).to(DEVICE) for _ in range(args.num_ints)]
    else:
        prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=args.oversample, num_det_px=args.num_det_px).to(DEVICE) for _ in range(args.num_ints)]

    
    # FIRST OPTIMIZE WITH THE REFERENCE IMAGE
    edge1, edge2 = 160 - int(args.num_det_px/2), 160 + int(args.num_det_px/2)

    real_im = np.load(f'{args.data_dir}/real_data/{args.reference_file}')[:1, edge1:edge2, edge1:edge2]
    real_im = real_im.astype(np.float32)
    plt.imsave(f'{vis_dir}/vis_measurement.png', real_im[0], cmap='viridis', origin='lower')
    reference = real_im
    reference = torch.from_numpy(reference).to(DEVICE)

    # print('YES reference median, sum ', reference.detach().median(), reference.detach().sum())
    if not args.no_median:
        ref_scaled = reference / reference.median()
    else: 
        ref_scaled = reference

    # print('YES ref_scaled median, sum ', ref_scaled.detach().median(), ref_scaled.detach().sum())

    # LOAD REFERENCE IMAGE TOO
    # scale by median before subtraction
    real_im = np.load(f'{args.data_dir}/real_data/{args.measurement_file}')[:1, edge1:edge2, edge1:edge2]
    observations = torch.from_numpy(real_im.astype(np.float32)).to(DEVICE)
    
    if not args.no_median:
        obs_scaled = observations / observations.median()
    else:
        obs_scaled = observations


    observations = torch.from_numpy(real_im.astype(np.float32)).to(DEVICE)
    
    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        pred = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred = torch.mean(torch.cat(pred, 0), 0)
    pred_np = pred.cpu().numpy()
    plt.imsave(f'{vis_dir}/vis_PSF_render_init.png', pred_np, cmap='viridis', origin='lower')


    if args.use_simulated_data:
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
            if not args.no_median:
                obs_scaled = observations / observations.median()
            else:
                obs_scaled = observations

            reference = torch.clone(sim[None]) 

            if not args.no_median:
                ref_scaled = reference / reference.median()
            else:
                ref_scaled = reference


    if not args.no_median:
        pred_scaled = pred / pred.median()
    else:
        pred_scaled = pred 

    
    est_residual = (obs_scaled - pred_scaled).detach().cpu().mean(0).numpy()
    print('est residual shape is ', est_residual.shape)
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')

    est_ref_residual = (obs_scaled - ref_scaled).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_ref_res_init.png', est_ref_residual, cmap='viridis', origin='lower')

    # scale by median before subtraction
    if not args.no_median:
        obs_scaled = observations / observations.median()
        pred_scaled = pred / pred.median()
    else:
        obs_scaled = observations
        pred_scaled = pred
    est_residual = (obs_scaled - pred_scaled).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')

    # Set up the optimizer and scheduler
    optics_params = list()
    """
    wfe_offsets: learns to offset the wfe_batch
    fpm_shifts: learns to shift the focal plane mask
    angle_offsets: learns to offset the star incident angle
    lyot_shifts: learns to shift the lyot mask
    nircam_offsets: learns to offset the nircam opd
    """
    for p_model in prop_models:
        optics_params += list(p_model.angle_offsets.parameters()) + list(p_model.nircam_offsets.parameters()) + list(p_model.wfe_offsets.parameters()) #+ list(p_model.flux_correction.parameters())
        if args.fit_everything:
            optics_params+= list(p_model.fpm_shifts.parameters()) + list(p_model.lyot_shifts.parameters()) + list(p_model.flux_correction.parameters())


    optimizer = torch.optim.AdamW(optics_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=args.lr)

    # specify the center region, used in the final loss function
    mask = torch.zeros_like(observations)
    # if not args.unenhance_center_loss:
    #     mask[..., 32:48, 32:48] = 1.0
    #     center_mask = mask > 0

    progress_arr_reference = []
    opd_vis_arr = []
    opd_vis_offset_arr = []
    if args.num_ints > 1:
        opd_vis_arr1 = []
        opd_vis_offset_arr_1 = []
        # opd_vis_arr2 = []
    nircam_opd_vis_arr = []
    nircam_opd_vis_arr_1 = []

    angles_offset_res = []
    angles_offset_res_1 = []

    residual_max_arr = []

    progress_arr_target = []

    cutoff_iter = args.ref_cutoff_iter

    tbar = tqdm.tqdm(range(args.iters + 1))
    for i in tbar:
        optimizer.zero_grad()

        pred_1 = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]

        # if i%100==0:
        #     print('YES pred_1 median, sum ', pred_1.detach().median(), pred_1.detach().sum())

        # compute loss in median-scaled space
        if not args.no_median:
            pred_scaled = pred_1 / pred_1.detach().median()
        else:
            pred_scaled = pred_1
        # if i%100==0:
        #     print('YES pred_scaled median, sum ', pred_scaled.detach().median(), pred_scaled.detach().sum())
        est_residual_obs = obs_scaled - pred_scaled.detach()

        est_residual_ref = ref_scaled - pred_scaled.detach()
        if i%100==0:
            with torch.no_grad():
                print('YES flux factor correction 1 is ', prop_models[0].flux_correction.forward(1.))
                # print('YES flux factor correction 2 is ', prop_models[1].flux_correction.forward(0.))
                print('YES flux angles 1 is ', prop_models[0].angle_offsets.forward())
                # print('YES flux angles 2 is ', prop_models[1].angle_offsets.forward())
                print('pred scaled max is ', pred_scaled.max())
                print('original data scaled is ', ref_scaled.max())

        global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled)
        obs_l1_loss = F.l1_loss(pred_scaled, obs_scaled)

        if i > cutoff_iter:
            if i < (args.stage_fluxpos_cutoff_iter + cutoff_iter):
                loss = obs_l1_loss
            else:
                loss = 0.5*obs_l1_loss
        else:
            loss = global_l1_loss

        if args.smooth is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])

            TVL = (total_variation_loss(opd1) + total_variation_loss(opd2))/2.

            nircam_opd = prop_models[0].nircam_offsets.get_res()
            nircam_op1 = prop_models[1].nircam_offsets.get_res()

            TVL += (total_variation_loss(nircam_opd) + total_variation_loss(nircam_op1))/2.

            if i > cutoff_iter:
                # loss = 0.001*obs_l1_loss
                loss = loss + args.smooth * TVL
            else:
                loss = loss + args.smooth * TVL



        if args.num_ints > 1 and args.OPD_loss_weight is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])

            # print('OPD is ', opd1)
            # print('OPD 2 is ', opd2)

            opd1_scaled = opd1 #/ opd1.detach().median()
            opd2_scaled = opd2 #/ opd2.detach().median()

            # print('OPD is scale ', opd1_scaled)
            # print('OPD 2 is scale', opd2_scaled)

            opds_disimilarity = F.smooth_l1_loss(opd1_scaled, opd2_scaled)

            # print('DISIMILARITY ', opds_disimilarity) 
            # print('BIG LOSS IS ', loss)

            if i > cutoff_iter:
                # loss = 0.001*obs_l1_loss
                loss = loss + args.OPD_loss_weight * opds_disimilarity * 1e15
            else:
                loss = loss + args.OPD_loss_weight * opds_disimilarity * 1e15

            # loss = loss + args.OPD_loss_weight * opds_disimilarity


        loss.backward()
        # for p_model in prop_models:
        #     for t in p_model.nircam_offsets.parameters():
        #         if t.requires_grad:
        #             t.grad *= 0.00
        if i > cutoff_iter:
            for p_model in prop_models:
                for t in p_model.wfe_offsets.parameters():
                    if t.requires_grad:
                        t.grad *= 1.0
            if i < (args.stage_fluxpos_cutoff_iter + cutoff_iter):
                with torch.no_grad():
                    tot_flux_pred = pred_scaled.sum()
                    tot_flux_target = obs_scaled.sum()
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
                # for t in p_model.nircam_offsets.parameters():
                #     if t.requires_grad:
                #         t.grad *= 0.01
        if i < args.stage_fluxpos_cutoff_iter:
            # Try manual fit of the flux, since optimization is being funny
            with torch.no_grad():
                tot_flux_pred = pred_scaled.sum()
                if i > cutoff_iter:
                    tot_flux_target = obs_scaled.sum()
                else:
                    tot_flux_target = ref_scaled.sum()

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


        optimizer.step()
        scheduler.step()

        if i > cutoff_iter:
            progress_arr_target.append(est_residual_obs[0])
        else:
            progress_arr_target.append(est_residual_obs[0])
            progress_arr_reference.append(est_residual_ref[0])

        
        if i % args.vis_freq == 0 and i > cutoff_iter:
            plt.imsave(f'{vis_dir}/vis_est_res_TARG_{i}.png', progress_arr_target[-1].cpu().numpy(), cmap='viridis', origin='lower')
        elif i % args.vis_freq == 0:
            plt.imsave(f'{vis_dir}/vis_est_res_REF_{i}.png', progress_arr_reference[-1].cpu().numpy(), cmap='viridis', origin='lower')

        # cur_opd = prop_models[0].wfe_offsets.get_res().squeeze().detach().cpu()
        # opd_vis_arr.append(cur_opd)
        # if args.num_ints > 1:
        #     cur_opd = prop_models[1].wfe_offsets.get_res().squeeze().detach().cpu()
        #     opd_vis_arr1.append(cur_opd)

        #     # cur_opd = prop_models[2].wfe_offsets.get_res().squeeze().detach().cpu()
        #     # opd_vis_arr2.append(cur_opd)

        # curr_nircam_opd = prop_models[0].nircam_offsets.get_res().squeeze().detach().cpu()
        # nircam_opd_vis_arr.append(curr_nircam_opd)
        # # residual_max_arr.append(est_residual.cpu().max().item())
        
        cur_opd = prop_models[0].wfe_offsets.forward(wfe_batch_list[0]).squeeze().detach().cpu()
        opd_vis_arr.append(cur_opd)

        opd_vis_offset_arr.append(prop_models[0].wfe_offsets.get_res().squeeze().detach().cpu())
        angles_offset_res.append(prop_models[0].angle_offsets())
        if args.num_ints > 1:
            cur_opd = prop_models[1].wfe_offsets.forward(wfe_batch_list[1]).squeeze().detach().cpu()
            opd_vis_arr1.append(cur_opd)

            angles_offset_res_1.append(prop_models[1].angle_offsets())

            opd_vis_offset_arr_1.append(prop_models[1].wfe_offsets.get_res().squeeze().detach().cpu())

            # cur_opd = prop_models[2].wfe_offsets.get_res().squeeze().detach().cpu()
            # opd_vis_arr2.append(cur_opd)
            curr_nircam_opd_1 = prop_models[1].nircam_offsets.get_res().squeeze().detach().cpu()
            nircam_opd_vis_arr_1.append(curr_nircam_opd_1)

        curr_nircam_opd = prop_models[0].nircam_offsets.get_res().squeeze().detach().cpu()
        nircam_opd_vis_arr.append(curr_nircam_opd)

        
        tbar_out = {'loss': global_l1_loss.item()}
        tbar.set_postfix(tbar_out)

    progress_arr = torch.stack(progress_arr_reference).cpu().numpy()[::5]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress_REFERENCE.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })
    
    psf_pixel_scale = 0.062424185
    target_numpys = np.squeeze(torch.stack(progress_arr_target).cpu().numpy())
    opd_vis_arr_numpys = np.squeeze(opd_vis_arr[-1].cpu().numpy())

    opd_vis_offset_arr_numpys = np.squeeze(opd_vis_offset_arr[-1].cpu().numpy())

    reference_numpys = np.squeeze(torch.stack(progress_arr_reference).cpu().numpy())

    angles_offset_res_numpys = np.squeeze(angles_offset_res[-1].detach().cpu().numpy())
    if args.num_ints > 1:
        opd_vis_arr_numpys1 = np.squeeze(opd_vis_arr1[-1].cpu().numpy())
        angles_offset_res_numpys_1 = np.squeeze(angles_offset_res_1[-1].detach().cpu().numpy())
        opd_vis_offset_arr_numpys_1 = np.squeeze(opd_vis_offset_arr_1[-1].cpu().numpy())
        # opd_vis_arr_numpys2 = np.squeeze(opd_vis_arr2[-1].cpu().numpy())
    nircam_opd_vis_arr_numpys = np.squeeze(nircam_opd_vis_arr[-1].cpu().numpy())
    if args.num_ints > 1:
        nircam_opd_vis_arr_1_numpys = np.squeeze(nircam_opd_vis_arr_1[-1].cpu().numpy())

    opd_vis_arr_numpys_initial = np.squeeze(opd_vis_arr[0].cpu().numpy())
    if args.num_ints > 1:
        opd_vis_arr_numpys1_initial = np.squeeze(opd_vis_arr1[0].cpu().numpy())
        # opd_vis_arr_numpys2 = np.squeeze(opd_vis_arr2[-1].cpu().numpy())
    nircam_opd_vis_arr_numpys_initial = np.squeeze(nircam_opd_vis_arr[0].cpu().numpy())

    np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',opd_vis_arr_numpys)
    np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_OFFSET_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',opd_vis_offset_arr_numpys)

    np.save(f'{vis_dir}/last_iteration_ANGLES_OFFSET_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',angles_offset_res_numpys)

    if args.num_ints > 1:
        np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_1.npy',opd_vis_arr_numpys1)
        np.save(f'{vis_dir}/last_iteration_ANGLES_OFFSET_oversample_{args.oversample}_wl_sampling_{args.num_wl}_1.npy',angles_offset_res_numpys_1)
        np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_OFFSET_oversample_{args.oversample}_wl_sampling_{args.num_wl}_1.npy',opd_vis_offset_arr_numpys_1)
        # np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_2.npy',opd_vis_arr_numpys2)
    np.save(f'{vis_dir}/last_iteration_NIRCAM_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',nircam_opd_vis_arr_numpys)
    if args.num_ints > 1:
        np.save(f'{vis_dir}/last_iteration_NIRCAM_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_1.npy',nircam_opd_vis_arr_1_numpys)
        

    np.save(f'{vis_dir}/first_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',opd_vis_arr_numpys_initial)
    if args.num_ints > 1:
        np.save(f'{vis_dir}/first_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_1.npy',opd_vis_arr_numpys1_initial)
        # np.save(f'{vis_dir}/last_iteration_ENTRANCE_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}_2.npy',opd_vis_arr_numpys2)
    np.save(f'{vis_dir}/first_iteration_NIRCAM_OPD_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',nircam_opd_vis_arr_numpys_initial)


    x_pos_real, y_pos_real = -6.5 * psf_pixel_scale, 11 * psf_pixel_scale # hand tuned for HIP 65426 in the pre-rotated frame
    # print('-------circular mask')
    # masked_residual = circular_mask(residual, x_pos_real, y_pos_real, 0.95, np.nan)
    # snr_list = []
    # for i in range(len(target_numpys)):
    #     curr_snr, _, _ = calc_snr(target_numpys[i], x_pos_real, y_pos_real,0.98,np.sqrt(x_pos_real**2 + y_pos_real**2),width=0.5)
    #     snr_list.append(curr_snr)
    
    # _, annulus, signal_blob = calc_snr(target_numpys[-1], x_pos_real, y_pos_real,0.98,np.sqrt(x_pos_real**2 + y_pos_real**2), width=0.5)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(annulus, origin='lower')

    # plt.subplot(122)
    # plt.imshow(signal_blob, origin='lower')

    # plt.tight_layout()
    # plt.savefig(f'{vis_dir}/target_final_snr_annulus.png')
    # plt.close()
    # peak_snr = np.nanmax(np.array(snr_list))
    # plt.figure()
    # plt.plot(snr_list)
    # plt.xlabel('Iterations')
    # plt.ylabel('SNR')
    # plt.grid()
    # plt.title(f'Oversample={args.oversample}, Wavelength sampling={args.num_wl}, max SNR = {peak_snr:.2f}')
    # plt.savefig(f'{vis_dir}/snrs_iterations.png')
    # plt.close()

    # np.save(f'{vis_dir}/snrs_iterations_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', np.array(snr_list))
    np.save(f'{vis_dir}/last_iteration_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', target_numpys[-1])
    np.save(f'{vis_dir}/last_iteration_REFERENCE_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', reference_numpys[-1])
    
    # np.save(f'{vis_dir}/max_snr_iteration_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', target_numpys[np.array(snr_list).argmax()])


    progress_arr = torch.stack(progress_arr_target).cpu().numpy()[::5]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress_TARGET.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })


    opd_vis_arr = torch.stack(opd_vis_arr)[::5]
    opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
    opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
    opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
    imageio.mimsave(f'{vis_dir}/opd_progress.mp4', opd_vis_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
    if args.num_ints > 1:
        opd_vis_arr = torch.stack(opd_vis_arr1)[::5]
        opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
        opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
        opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
        imageio.mimsave(f'{vis_dir}/opd_progress_1.mp4', opd_vis_arr, 
                        'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
        
        # opd_vis_arr = torch.stack(opd_vis_arr2)[::5]
        # opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
        # opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
        # opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
        # imageio.mimsave(f'{vis_dir}/opd_progress_2.mp4', opd_vis_arr, 
        #                 'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
    
    # opd_vis_arr = torch.stack(nircam_opd_vis_arr)[::5]
    # opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
    # opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
    # opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
    # imageio.mimsave(f'{vis_dir}/nircam_opd_progress.mp4', opd_vis_arr, 
    #                 'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
