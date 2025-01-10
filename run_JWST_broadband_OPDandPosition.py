import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F

from dl_utils import pixel_coords, crop_to, arcsec2rad, partial_MFT


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

class Zernike(nn.Module):
    def __init__(self, n_max: int, m_max: int, npixels: int):
        super().__init__()
        self.n_max = n_max
        self.m_max = m_max
        self.npixels = npixels
        number_of_zernikes = (n_max + 1) * (n_max + 2) // 2
        self.basis_function_weights = nn.Parameter(
            torch.randn(number_of_zernikes, dtype=torch.float64), requires_grad=True
        )
        self.center_x = nn.Parameter(
            torch.tensor(npixels / 2, dtype=torch.float64), requires_grad=False
        )
        self.center_y = nn.Parameter(
            torch.tensor(npixels / 2, dtype=torch.float64), requires_grad=False
        )

    @staticmethod
    def _factorial(x: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(x + 1).exp()

    def _f_nms(self, n: int, m: int, s: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        s_3d = s.view(-1, 1, 1)
        return (
            (-1) ** s_3d
            * self._factorial(n - s_3d)
            * self._factorial((n - m) / 2 - s_3d)
            * rho ** (n - 2 * s_3d)
            / (self._factorial(s_3d) * self._factorial((n + m) / 2 - s_3d))
        )

    def _summation(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        indices_s = torch.arange(0, (n - m) / 2, dtype=torch.float64, device=rho.device)
        return self._f_nms(n, m, indices_s, rho).sum(dim=0)

    def _R_nm(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        radial_contribution = torch.zeros_like(rho, dtype=torch.float64)
        radial_contribution += self._summation(n, m, rho)
        return radial_contribution

    def sum_basis_funcs(self):
        device = self.center_x.device
        x = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64, device=device)
        y = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64, device=device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        x = x - self.center_x
        y = y - self.center_y
        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        idx = 0
        output = torch.zeros((self.npixels, self.npixels), dtype=torch.float64,device=device)

        for n in range(self.n_max + 1):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    if m == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sqrt(torch.tensor(n + 1, dtype=torch.float32, device=output.device))
                        )
                        idx += 1
                    elif m % 2 == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.cos(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32, device=output.device))
                        )
                        idx += 1
                    else:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sin(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32, device=output.device))
                        )
                        idx += 1

        return output

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
                self.amplitude = self.basis.sum_basis_funcs()
            else:
                self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasor(self, angles_offset=None):
        opd = self.get_tilt_opd(angles_offset)
        if self.basis is not None:
            self.amplitude = self.basis.sum_basis_funcs()
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
    wavefronts_list1 = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_STAR, basis=Zernike(20, 10, wf_npix)).to(DEVICE) for wl in wlen_weights[0]]

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
    plt.imsave(f'{vis_dir}/vis_PSF_render_init.png', pred_np, cmap='viridis', origin='lower')

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

    tbar = tqdm.tqdm(range(args.iters + 1))
    for i in tbar:
        optimizer.zero_grad()

        pred_1 = [p_model(wavefronts_list1, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models]
        pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]

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

    progress_arr = torch.stack(progress_arr).cpu().numpy()[::5]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })

    opd_vis_arr = torch.stack(opd_vis_arr)[::5]
    opd_vis_arr = (opd_vis_arr - opd_vis_arr[0:1]).abs().numpy()
    opd_vis_arr = (opd_vis_arr - opd_vis_arr.min()) / (opd_vis_arr.max() - opd_vis_arr.min())
    opd_vis_arr = np.uint8(cm.coolwarm(opd_vis_arr) * 255)
    imageio.mimsave(f'{vis_dir}/opd_progress.mp4', opd_vis_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'fps': 30, })
