import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from poppy.zernike import zernike_basis
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    #rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=28)

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F

from dl_utils import pixel_coords, crop_to, arcsec2rad, partial_MFT, crop_image, compute_zernike_basis


def load_data(data_dir):
    basic_masks_dir = os.path.join(data_dir, 'masks_1024')

    aperture = np.load(f'{basic_masks_dir}/primary_transmission_1024.npy')
    aperture = np.flip(aperture, axis=0)
    lyot = np.load(f'{basic_masks_dir}/circlyotstop_transmission_1024.npy')

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


class AngleOffsetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self):

        return self.data

class ZernikeNet(nn.Module):
        def __init__(self, PSF_size, hidden_dim=32, phs_layers=2):
            super(ZernikeNet, self).__init__()
            self.PSF_size = PSF_size
            self.basis = nn.Parameter(compute_zernike_basis(
                num_polynomials=28,
                field_res=(PSF_size, PSF_size)).permute(1, 2, 0),
                requires_grad=False)

            hidden_dim = hidden_dim
            in_dim = self.basis.shape[-1]

            act_fn = nn.LeakyReLU(inplace=True)
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(phs_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(act_fn)

            layers.append(nn.Linear(hidden_dim, 1))
            self.wavefront = nn.Sequential(*layers)

        def forward(self, x, y):
            basis_at_coordinate = self.basis[y, x, :]
            return self.wavefront(basis_at_coordinate).squeeze(-1)


class Wavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: float, peak_flux: float, angles = None, basis=None):
        super().__init__()
        self.wavelength = nn.Parameter(torch.from_numpy(np.asarray(wavelength, float)), requires_grad=False)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumber = 2 * np.pi / self.wavelength
        self.npixels = npixels
        self.diameter = diameter
        self.peak_flux = peak_flux
        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles, requires_grad=True)
        self.basis = basis
        self.reset()
    
    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            if self.basis is not None:
                self.amplitude = torch.zeros(self.npixels, self.npixels, device=self.coordinates.device)
                i_indices, j_indices = torch.meshgrid(torch.arange(self.npixels), torch.arange(self.npixels), indexing='ij')
                i_indices = i_indices.flatten()
                j_indices = j_indices.flatten()
                amplitudes = self.basis(j_indices, i_indices)
                amplitudes = amplitudes.view(1, self.npixels, self.npixels)
                self.amplitude = nn.Parameter(self.amplitude.to(DEVICE), requires_grad=True)
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

        #print(phasor.shape)
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

    def forward_fpm(self, phasor, layer):
        npixels_in = self.npixels
        phasor = self.propagate(phasor, pad=1)
        new_phasor = phasor * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfe(self, phasor, layer, wlen):
        wfe = torch.exp(1j * layer * 2 * np.pi / wlen)
        new_phasor = phasor * wfe
        return new_phasor


class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, args):
        super().__init__()
        self.aperture = aperture
        self.lyot = lyot
        self.fpm = fpm
        self.nircam_opd = nircam_opd

        self.lyot_shifts = ShiftModule(lyot.shape[-2], lyot.shape[-1])
        self.fpm_shifts = ShiftModule(fpm.shape[-2], fpm.shape[-1])
        self.nircam_offsets = GridOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])
        self.wfe_offsets = OPDOffsetModule(fpm.shape[-2], fpm.shape[-1])
        self.angle_offsets = AngleOffsetModule()

        (npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse) = args
        xmats, ymats, mults = [], [], []
        for i in range(len(wavelengths)):
            args = (npixels, wavelengths[i], true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)
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
            phasor_fpm = wavefront.forward_fpm(phasor_ap, self.fpm_shifts(self.fpm[:, i]))
            phasor_lyot = wavefront.forward(phasor_fpm, self.lyot_shifts(self.lyot))
            phasor_nircam_opd = wavefront.forward_wfe(phasor_lyot, self.nircam_offsets(self.nircam_opd[:, i]), wavelenghts[i])
            phasor = (self.y_mat[i].T @ phasor_nircam_opd) @ self.x_mat[i]
            phasor *= self.mult[i]
            w = wavefront.peak_flux ** 0.5
            out = (torch.abs(phasor) * w) ** 2

            if output is None:
                output = out * wl_weights[i]
            else:
                output += out * wl_weights[i]

        output = torch.flip(output, dims=(-2,))

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
    parser.add_argument('--scene_name', default='HIP65426', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--vis_freq', default=50, type=int)
    parser.add_argument('--lr', default=1e-8, type=float)
    parser.add_argument('--star_offset_x', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--star_offset_y', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--basis', default='zernike', help='Basis for the wavefront', type=str)
    args = parser.parse_args()

    DEVICE = 'cuda'

    # Number of pixels in the wavefront
    wf_npix = 1024
    # Diameter of the aperture
    diameter = 6.56
    # Number of pixels in the PSF
    psf_npix = 80
    # Pixel scale in our detector, in arcseconds. For our case, around 14 miliarcseconds per pixel is expected.
    psf_pixel_scale = 0.062424185

    # Load data from args.data_dir
    aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=args.data_dir)

    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]
    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights)

    sampledWFEs = np.load(f'{args.data_dir}/masks_1024/opd_20220730.npy')
    sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch = sampledWFEs.contiguous().to(DEVICE)

    ############
    # Set up export directories
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}/{args.exp_name}'
    os.makedirs(vis_dir, exist_ok=True)

    contrast_normalization = 0.003716380479003919
    sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data; VERY ROUGH! To be fixed
    photons_normalization = 90578.00102527262 * sim_to_real_scaling # mJy/sr
    peak_flux_star  = photons_normalization / contrast_normalization

    offset_STAR = nn.Parameter(torch.FloatTensor([args.star_offset_x * arcsec2rad(1 / (psf_pixel_scale * 1000)), args.star_offset_y * arcsec2rad(1 / (psf_pixel_scale * 1000))]))
    
    # Set up the wavefront objects
    if args.basis == 'zernike':
        wavefronts_list1 = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_STAR, basis=ZernikeNet(wf_npix)).to(DEVICE) for wl in wlen_weights[0]]
    else:
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
    prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args).to(DEVICE) for _ in range(1)]

    real_im = np.load(f'{args.data_dir}/real_data/{args.measurement_file}')[:1, 120:200, 120:200]
    plt.imsave(f'{vis_dir}/vis_measurement.png', real_im[0], cmap='viridis', origin='lower')
    observations = torch.from_numpy(real_im).to(DEVICE)
    
    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        pred = [p_model(wavefronts_list1, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models]
        pred = torch.mean(torch.cat(pred, 0), 0)
    pred_np = pred.cpu().numpy()
    #print(pred_np.shape)
    plt.imsave(f'{vis_dir}/vis_PSF_render_init.png', pred_np, cmap='viridis', origin='lower')

    z_score_measurement = (real_im[0] - real_im[0].mean()) / real_im[0].std()
    plt.imsave(f'{vis_dir}/vis_measurement_zscore.png', z_score_measurement, cmap='viridis', origin='lower')

    z_score_pred = (pred_np - pred_np.mean()) / pred_np.std()
    plt.imsave(f'{vis_dir}/vis_PSF_render_zscore_init.png', z_score_pred, cmap='viridis', origin='lower')

    # scale by median before subtraction
    obs_scaled = observations / observations.median()
    pred_scaled = pred / pred.median()
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
        #optics_params += list(p_model.angle_offsets.parameters()) + list(p_model.nircam_offsets.parameters()) + list(p_model.wfe_offsets.parameters())
        optics_params += list(p_model.angle_offsets.parameters()) + list(p_model.nircam_offsets.parameters())

    optimizer = torch.optim.AdamW(optics_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=args.lr)

    # specify the center region, used in the final loss function
    mask = torch.zeros_like(observations)
    mask[..., 32:48, 32:48] = 1.0
    center_mask = mask > 0

    progress_arr = []
    opd_vis_arr = []
    residual_max_arr = []
    zscore_arr = []

    tbar = tqdm.tqdm(range(args.iters + 1))
    for i in tbar:
        optimizer.zero_grad()

        pred_1 = [p_model(wavefronts_list1, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models]
        pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]
        z_score_pred = (pred_1.cpu().detach().numpy() - pred_1.cpu().detach().numpy().mean()) / pred_1.cpu().detach().numpy().std()
        zscore_arr.append(z_score_pred)

        # compute loss in median-scaled space
        pred_scaled = pred_1 / pred_1.detach().median()
        
        center_loss = F.smooth_l1_loss(pred_scaled[center_mask], obs_scaled[center_mask])
        global_l1_loss = F.smooth_l1_loss(pred_scaled, obs_scaled)
        loss = global_l1_loss + center_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        est_residual = obs_scaled - pred_scaled.detach()
        progress_arr.append(est_residual[0])
        if i % args.vis_freq == 0 and i > 0:
            plt.imsave(f'{vis_dir}/vis_est_res_{i}.png', progress_arr[-1].cpu().numpy(), cmap='viridis', origin='lower')

        cur_opd = prop_models[0].wfe_offsets.get_res().squeeze().detach().cpu()
        opd_vis_arr.append(cur_opd)
        residual_max_arr.append(est_residual.cpu().max().item())

        tbar_out = {'loss': global_l1_loss.item()}
        tbar.set_postfix(tbar_out)

    if args.basis != 'zernike':
        final_wavefront_parameters = [parameters_to_vector(wavefront.parameters()) for wavefront in wavefronts_list1]
        final_params_size = final_wavefront_parameters[0].shape[0]
        print(f'Variance Final Parameters: {torch.var(final_wavefront_parameters[0])}')
        variance_final_params = torch.var(final_wavefront_parameters[0])

        random_params_matrix = torch.zeros(final_params_size, len(final_wavefront_parameters)).to(DEVICE)
        random_two_params_matrix = torch.zeros(final_params_size, len(final_wavefront_parameters)).to(DEVICE)
        for i in range(len(final_wavefront_parameters)):
            random_params = torch.randn(final_params_size).to(DEVICE)
            random_two_params = torch.randn(final_params_size).to(DEVICE)
            orthogonal_projection = torch.dot(random_two_params, random_params) / torch.dot(random_params, random_params)
            random_two_params = random_two_params - orthogonal_projection * random_params
            random_params_matrix[:, i] = random_params
            random_two_params_matrix[:, i] = random_two_params


        bound = torch.max(torch.stack([torch.mean(final_wavefront_parameters[i]) + 2 * torch.std(final_wavefront_parameters[i]) for i in range(len(final_wavefront_parameters))])
    )


        loss_landscape = torch.zeros(100, 100)
        loss_landscape_x = torch.linspace(-bound, bound, 100)
        loss_landscape_y = torch.linspace(-bound, bound, 100)

        
        for i in tqdm.tqdm(range(len(loss_landscape_x)), desc="Outer Loop"):
            for j in range(len(loss_landscape_y)):
                for k in range(len(final_wavefront_parameters)):
                    random_params = random_params_matrix[:, k]
                    random_two_params = random_two_params_matrix[:, k]
                    vector_to_parameters(final_wavefront_parameters[k] + loss_landscape_x[i] * random_params + loss_landscape_y[j] * random_two_params, wavefronts_list1[k].parameters())

                pred_1 = [p_model(wavefronts_list1, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models]
                pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]
                pred_scaled = pred_1 / pred_1.detach().median()
                center_loss = F.smooth_l1_loss(pred_scaled[center_mask], obs_scaled[center_mask])
                global_l1_loss = F.smooth_l1_loss(pred_scaled, obs_scaled)
                loss = global_l1_loss + center_loss
                loss_landscape[i, j] = loss.item()
        
        loss_landscape = loss_landscape.cpu().numpy()
        np.save(f'{vis_dir}/loss_landscape_100.npy', loss_landscape)

    progress_arr = torch.stack(progress_arr).cpu().numpy()[::5]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })

    zscore_arr = np.array(zscore_arr)
    final_zscore = zscore_arr[-1]
    plt.imsave(f'{vis_dir}/final_zscore_psf.png', final_zscore[0], cmap='viridis', origin='lower')

    nterms = 30
    npix = psf_npix
    outside = np.nan 
    basis = zernike_basis(nterms=nterms, npix=npix, outside=outside)
    z_weights_real = []
    z_weights_psf = []
    for i, zernike_mode in enumerate(basis, start=1):
        zernike_nan_to_zero = np.nan_to_num(zernike_mode)
        z_weight_real = (np.nansum(zernike_nan_to_zero * z_score_measurement) / np.nansum(zernike_nan_to_zero * zernike_nan_to_zero))
        z_weight_psf = (np.nansum(zernike_nan_to_zero * final_zscore) / np.nansum(zernike_nan_to_zero * zernike_nan_to_zero))
        z_weights_real.append(z_weight_real)
        z_weights_psf.append(z_weight_psf)

        real_projection = z_weight_real * zernike_mode
        psf_projection = z_weight_psf * zernike_mode

        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 5), tight_layout=True)
        vmin = min(np.nanmin(real_projection), np.nanmin(psf_projection))
        vmax = max(np.nanmax(real_projection), np.nanmax(psf_projection))
        real_image = ax[0].imshow(real_projection, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[0].set_title(f'Real Projection {i}')
        colorbar_one = fig.colorbar(real_image, cax=cax)
        psf_image = ax[1].imshow(psf_projection, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[1].set_title(f'PSF Projection {i}')
        colorbar_two = fig.colorbar(psf_image, cax=cax)
        diff_image = ax[2].imshow(real_projection - psf_projection, cmap='viridis', origin='lower')

        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[2].set_title('Difference')
        colorbar_three = fig.colorbar(diff_image, cax=cax)
        plt.savefig(f'{vis_dir}/final_zscore_projection_{i}.png')
        plt.close()

    plt.figure(figsize=(18, 10))
    plt.plot(z_weights_real, label='Real')
    plt.plot(z_weights_psf, label='PSF')
    plt.plot(np.array(z_weights_real) - np.array(z_weights_psf), label='Diff')
    plt.plot(np.zeros(len(z_weights_real)), 'k--')
    plt.xlabel('Zernike Mode')
    plt.ylabel('Weight')
    plt.legend()
    plt.savefig(f'{vis_dir}/zernike_weights.png')
    plt.close()

    opd_vis_arr = torch.stack(opd_vis_arr)[::5]
    opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
    opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
    opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
    imageio.mimsave(f'{vis_dir}/opd_progress.mp4', opd_vis_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
