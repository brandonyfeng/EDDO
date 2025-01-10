import os
import tqdm
import imageio
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F

from dl_utils import pixel_coords, crop_to, circle, arcsec2rad, partial_MFT

DEVICE = 'cuda'
ffmpeg_kargs = {'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 4, }


def load_data(N, data_dir, opd_data_dir, multi_snap):
    aperture = np.load(f'{data_dir}/primary_transmission_1024.npy')
    aperture = np.flip(aperture, axis=0) # Flipping to make aperture coincide with Lyot
    lyot = np.load(f'{data_dir}/circlyotstop_transmission_1024.npy')
    fpm = np.load(f'{data_dir}/transmission_MASK335R_4_5um_2048.npy')
    nircam_opd = np.load(f'{data_dir}/FDA_NIRCamLWA_opd_1024.npy')

    aperture_opd_list = []
    for opd_file in sorted(os.listdir(opd_data_dir)):
        if opd_file.endswith('.npy'):
            opd = np.load(os.path.join(opd_data_dir, opd_file))
            opd = np.flip(opd, axis=0) # Flipping to make aperture coincide with Lyot
            aperture_opd_list.append(opd)
    aperture_opd_list = np.array(aperture_opd_list)

    sampledWFEs = aperture_opd_list[0:1]
    sampledWFEs_2 = aperture_opd_list[1:2]

    if N > 1:
        sampledWFEs = np.repeat(sampledWFEs, N, axis=0)
        sampledWFEs_2 = np.repeat(sampledWFEs_2, N, axis=0)

    if N > 1 and multi_snap:
        delta = sampledWFEs_2 - sampledWFEs
        increment = np.linspace(0.5, 1, N)
        sampledWFEs_2 = sampledWFEs + delta * increment[:, None, None]

    return aperture, lyot, fpm, nircam_opd, sampledWFEs, sampledWFEs_2

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
        indices_s = torch.arange(0, (n - m) / 2, dtype=torch.float64)
        return self._f_nms(n, m, indices_s, rho).sum(dim=0)

    def _R_nm(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        radial_contribution = torch.zeros_like(rho, dtype=torch.float64)
        radial_contribution += self._summation(n, m, rho)
        return radial_contribution

    def sum_basis_funcs(self):
        x = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64)
        y = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64)
        x, y = torch.meshgrid(x, y, indexing="xy")
        x = x - self.center_x
        y = y - self.center_y
        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        idx = 0
        output = torch.zeros((self.npixels, self.npixels), dtype=torch.float64)

        for n in range(self.n_max + 1):
            for m in range(n + 1)):
                if (n - m) % 2 == 0:
                    if m == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sqrt(torch.tensor(n + 1, dtype=torch.float32))
                        )
                        idx += 1
                    elif m % 2 == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.cos(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32))
                        )
                        idx += 1
                    else:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sin(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32))
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
                                

    def get_phasor(self):
        if self.basis is not None:
            self.amplitude = self.basis.sum_basis_funcs()
        opd = self.get_tilt_opd()
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilt_opd(self):
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
        phasor = self.propagate(phasor, pad=2)
        new_phasor = phasor * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfe(self, phasor, layer):
        wfe = torch.exp(1j * layer * self.wavenumber)
        new_phasor = phasor * wfe
        return new_phasor


class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, *args):
        super().__init__()
        self.aperture = aperture
        self.lyot = lyot
        self.fpm = fpm
        self.nircam_opd = nircam_opd
        x_mat, y_mat, mult = partial_MFT(*args)
        self.x_mat = nn.Parameter(x_mat, requires_grad=False)
        self.y_mat = nn.Parameter(y_mat, requires_grad=False)
        self.mult = nn.Parameter(torch.tensor(mult, dtype=torch.float64), requires_grad=False)

    def forward(self, wavefront, wfe):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward_wfe(phasor, wfe)
        phasor = wavefront.forward(phasor, self.aperture,normalize=True)
        phasor = wavefront.forward_fpm(phasor, self.fpm)
        phasor = wavefront.forward(phasor, self.lyot)
        phasor = wavefront.forward_wfe(phasor, self.nircam_opd)

        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.peak_flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out

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
    parser.add_argument('--data_dir', default='./FilesJWSTMasks/masks_1024', type=str)
    parser.add_argument('--opd_data_dir', default='./OPDs_JWST_dates', type=str)
    parser.add_argument('--scene_name', default='guidestar', type=str)
    parser.add_argument('--num_t', help='Number of measurements', default=1, type=int)
    parser.add_argument('--contrast_mult', default=1, type=float)
    parser.add_argument('--contrast_exp', default=-5, type=int)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--iters', default=200, type=int)
    parser.add_argument('--lr', default=1e-10, type=float)
    parser.add_argument('--offset_r', default=0.6, type=float)
    parser.add_argument('--offset_t', default=0.3, type=float)
    parser.add_argument('--shot_noise', default=1.0, type=float, help='Shot noise level')
    parser.add_argument('--read_noise', default=42.0, type=float, help='Read noise level')
    parser.add_argument('--loss_fn', default='L1', type=str)
    parser.add_argument('--drift_ratio', default=1.0, type=float)
    parser.add_argument('--log_progress', action='store_true')
    parser.add_argument('--multi_snapshot', action='store_true')

    args = parser.parse_args()

    # Planet contrast
    planet_contrast = args.contrast_mult * (10 ** args.contrast_exp)
    # Ratio of error to add to the WFE
    drift_ratio = args.drift_ratio
    # Number of measurements
    N = args.num_t
    # Number of iterations
    iters = args.iters
    # Learning rate
    lr = args.lr
    # Visualization frequency (in iterations)
    vis_freq = args.vis_freq
    # epsilon added to log to avoid log(0)
    vis_norm_eps = 1e-3

    # Wavelength
    wavelen = 4.5e-6
    # Number of pixels in the wavefront
    wf_npix = 1024
    # Diameter of the aperture
    diameter = 6.56
    # Number of pixels in the PSF
    psf_npix = 80
    # Pixel scale in our detector, in arcseconds. For our case, around 14 miliarcseconds per pixel is expected.
    psf_pixel_scale = 0.062424185
    # Radius of the PSF in pixels
    psf_radius = (psf_npix * arcsec2rad(psf_pixel_scale)) / 2
    # Offset of the planet from the center of the PSF
    #offset = torch.FloatTensor([args.offset_x, args.offset_y]).view(2)    
    coord_r, coord_t = args.offset_r, args.offset_t * np.pi
    coord_x = arcsec2rad(coord_r * np.cos(coord_t))
    coord_y = arcsec2rad(coord_r * np.sin(coord_t))
    offset = torch.FloatTensor([coord_x, coord_y]).view(2)

    # Load data from args.data_dir
    aperture, lyot, fpm, nircam_OPD, sampledWFEs, sampledWFEs_2 = load_data(N=N, data_dir=args.data_dir, opd_data_dir=args.opd_data_dir, multi_snap=args.multi_snapshot)

    coords = pixel_coords(wf_npix, diameter)
    aperture = circle(coords, 0.5 * diameter).to(DEVICE)[None]
    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    sampledWFEs = torch.tensor(sampledWFEs).float()
    sampledWFEs = F.interpolate(sampledWFEs.unsqueeze(1), size=(wf_npix, wf_npix), mode='bilinear').squeeze()

    sampledWFEs_2 = torch.tensor(sampledWFEs_2).float()
    sampledWFEs_2 = F.interpolate(sampledWFEs_2.unsqueeze(1), size=(wf_npix, wf_npix), mode='bilinear').squeeze()

    if len(sampledWFEs.shape) == 2:
        sampledWFEs = sampledWFEs.unsqueeze(0).repeat(2, 1, 1)

    a, b = sampledWFEs[:1], sampledWFEs[1:2]
    sampledWFEs = torch.cat([a * e + b * (1 - e) for e in np.linspace(0, 1, N)])
    sampledWFEs = sampledWFEs.to(DEVICE)
    sampledWFEs_2 = sampledWFEs_2.to(DEVICE)

    ############
    # Set up export directories
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}/N{args.num_t}/{args.contrast_mult}x10_{args.contrast_exp}_err_{drift_ratio}_off_{args.offset_r}_{args.offset_t}'

    if args.loss_fn == 'L2':
        vis_dir += '_L2_loss'
    if args.shot_noise > 0:
        vis_dir += f'_shot_noise_{args.shot_noise}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # NEW PHOTOMETRIC CALIBRATION: Based on a 2 hour exposure of the star Beta Pictoris, using its W2 apparent magnitude
    # (i.e. its brightness at 4.6 microns, which is the wavelength/filter we are interested in) from the WISEA catalog:
    # https://vizier.cds.unistra.fr/viz-bin/VizieR-S?WISEA%20J054717.10-510358.4
    # And using the zero point of the F444W filter in the NIRCam instrument of JWST.

    # Using photometry data from the JWST files as well, we use the conversion of 1 count per second ~ 2.51 MJy/sr
    # Since we are working at 4.5 microns, we assume a detector gain of 1.84 and a quantum efficiency of 80%.

    # All conversions done using the peak pixel of the simulations and the peak pixel of real data,
    # so we do the same here. The "flux" parameter was changed to "peak flux" and is used in this way.

    # Brightness of the peak pixel of an unocculted (e.g. no focal plane mask) star;
    # dividing the image by this gives by definition the image in contrast units
    # if the image is simulated the way it is simulated here
    # (i.e. total intensity in aperture = 1, and electric field is not
    # normalized when passing through each mask to capture the real throughput).
    contrast_normalization = 0.003716380479003919

    # From calibration with stellar photometry described above, this is the number of photons
    # corresponding to a contrast = 1 for a 2 hour observation of Beta Pictoris (computed from
    # the zero point of F444W in JWST's NIRCam and Beta Pictoris' brightness or magnitude in that filter).
    # So multiplying an image in contrast units by this amount gives the image in photons.
    photons_normalization = 10334325889.379316
    
    # IMPORTANT CHANGE:
    # Now we compute the peak fluxes. These are the peak fluxes we would get if we
    # observe the sources without the focal plane mask. This is what we want since it more
    # realistically captures the resulting flux at the end of the detector, now that our
    # model captures the true optical throughput.
    peak_flux_star  = photons_normalization / contrast_normalization

    # VERY IMPORTANT CHANGE:
    # Same thing for the planet here. One important consequence is that now
    # the desired contrast is set to be the true contrast of the planet to the star,
    # if we were to see the planet without the focal plane mask. This is important because:
    #     1) This is the way "contrast" is reported in the literature.
    #     2) If the planet is closer to the focal plane mask (e.g., within like 0.8 arcseconds)
    #        the planet itself can be partially occulted too by the mask, resulting in the planet 
    #        appearing at the detector *fainter* than the true contrast.
    #     3) For planets far from the focal plane mask (e.g., more than 1 arcsecond) the occulting
    #        is almost negligible, and the planet appears at the detector with pretty much the same
    #        brightness (contrast) as the true value. For planets closer in, the occulting is not
    #        negligible, but as close as 0.6 arcseconds (the inner working angle)
    #        the difference is only like a factor of 2. So in our scales, our dynamic range 
    #        (orders of magnitude) is large enough that the difference in performance will not 
    #        be too dramatic.
    #        But for even closer in planets, the occulting is more severe, and it causes a change in
    #        measured contrast of one order of magnitude or more for planets inside ~0.4 arcseconds.
    #        So if any planet is tested inside this range, it must be done in this new way to account
    #        for how fainter (than its true brightness) the planet would appear to us in real life in 
    #        that position of the field of view.
    # This new way of simulating the planet allows way more realistic simulations, since (in short)
    # it includes more effects of the field-of-view dependence on the throughput of the planet.

    peak_flux_planet =  planet_contrast * photons_normalization / contrast_normalization
    flux_1, flux_2 = peak_flux_star, peak_flux_planet

    # Set up the wavefront objects
    plane_wave_1 = Wavefront(wf_npix, diameter, wavelen, flux_1, basis=Zernike(20, 10, wf_npix))
    plane_wave_1 = plane_wave_1.to(DEVICE)
    plane_wave_2 = Wavefront(wf_npix, diameter, wavelen, flux_2, offset, basis=Zernike(20, 10, wf_npix))
    plane_wave_2 = plane_wave_2.to(DEVICE)

    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = wf_npix
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wavelen, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Set up the propagation objects
    p1_model = PointPropagate(aperture, lyot, fpm, nircam_OPD, *prop_args)
    p2_model = PointPropagate(aperture, lyot, fpm, nircam_OPD, *prop_args)
    p1_model = p1_model.to(DEVICE)
    p2_model = p2_model.to(DEVICE)

    # Visualize the resulting PSF
    out_1, out_2 = [], []
    out_1 = p1_model(plane_wave_1, sampledWFEs).detach().cpu()
    out_2 = p2_model(plane_wave_2, sampledWFEs).detach().cpu()
    out = out_1 + out_2

    # Visualize subtracted PSF using the ground truth WFE
    est_residual = out - out_1
    est_residual = est_residual.detach().cpu().numpy()
    est_residual_log10 = np.log10(est_residual + vis_norm_eps)
    est_residual_log10 = est_residual_log10 / est_residual_log10.max()
    est_residual_log10 = np.uint8(cm.viridis(est_residual_log10) * 255)
    imageio.imsave(f'{vis_dir}/vis_est_log10_residual_gt.png', est_residual_log10[0])
    est_residual = est_residual / est_residual.max()
    est_residual = np.uint8(cm.viridis(est_residual) * 255)
    imageio.imsave(f'{vis_dir}/vis_residual_gt.png', est_residual[0])

    # Find planet location in image pixels
    planet_coords = (offset / psf_pixel_scale) + (psf_npix / 2)
    planet_coords = planet_coords.int().cpu().numpy()
    planet_coords = [planet_coords[1], planet_coords[0]]

    # Plot the plantet location on a grid
    image = np.zeros((psf_npix, psf_npix))
    image[planet_coords[0], planet_coords[1]] = 1
    plt.imsave(f'{vis_dir}/planet_location.png', image, cmap='gray')

    # crop the residual PSF around the planet location
    begin_x, end_x = planet_coords[0] - 5, planet_coords[0] + 5
    begin_y, end_y = planet_coords[1] - 5, planet_coords[1] + 5

    if args.shot_noise > 0:
        shot_noise = torch.normal(mean = out, std = args.shot_noise * torch.sqrt(out)) - out
        read_noise = torch.normal(mean = 0, std = args.read_noise * torch.ones_like(out))
        noise = shot_noise + read_noise

        out_noisy = out + noise
        out = torch.clamp(out_noisy, min=0.0)
        planet = out_2
        star = out_1

        x = np.array([f'Star: min={star.min()}, max={star.max()}', f'Planet: min={planet.min()}, max={planet.max()}', f'Noise: min={noise.min()}, max={noise.max()}'])
        np.savetxt(f'{vis_dir}/photon_stats.txt', x, fmt='%s')

        # Visualize subtraction result
        est_residual = torch.clamp(out - out_1, min=0.0)
        est_residual = est_residual.detach().cpu().numpy()
        est_residual_log10 = np.log10(est_residual + vis_norm_eps)
        est_residual_log10 = est_residual_log10 / est_residual_log10.max()
        est_residual_log10 = np.uint8(cm.viridis(est_residual_log10) * 255)
        imageio.imsave(f'{vis_dir}/vis_res_log10_gt_w_noise.png', np.mean(est_residual_log10, axis=0).astype(np.uint8))
        est_residual = est_residual / est_residual.max()
        est_residual = np.uint8(cm.viridis(est_residual) * 255)
        imageio.imsave(f'{vis_dir}/vis_res_gt_w_noise.png', np.mean(est_residual, axis=0).astype(np.uint8))

    observations = out.detach().contiguous().to(DEVICE)
    wfe_gt = sampledWFEs
    wfe_delta = sampledWFEs_2 - sampledWFEs

    # Add error to the WFE
    wfe_off = drift_ratio * wfe_delta
    wfe_batch = sampledWFEs + wfe_off
    wfe_batch = wfe_batch.contiguous()
    wfe_batch.requires_grad = True

    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        e_pred = []
        for batch in range(0, N, 32):
            cur_wfe = wfe_batch[batch:batch + 32]
            pred_1 = p1_model(plane_wave_1, cur_wfe)
            e_pred.append(pred_1.detach().cpu())
        e_pred = torch.cat(e_pred)

    est_residual = torch.relu(out - e_pred)
    est_residual = est_residual.detach().cpu().numpy()
    est_residual_normed = est_residual / est_residual.max()
    est_residual_normed = np.uint8(cm.viridis(est_residual_normed) * 255)
    imageio.imsave(f'{vis_dir}/vis_est_res_init.png', np.mean(est_residual_normed, axis=0).astype(np.uint8))

    init_est_star_psf = e_pred.detach().cpu().numpy()
    init_est_residual_psf = est_residual.copy()

    # Record the initial losses
    wfe_err, img_err = F.l1_loss(wfe_batch, wfe_gt), F.l1_loss(e_pred, observations.cpu())
    img_loss_arr, wfe_err_arr = [], []
    img_loss_arr.append(img_err.item())
    wfe_err_arr.append(wfe_err.item())

    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam([wfe_batch], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-20)
    loss_scale = init_est_star_psf.max()

    if args.log_progress:
        progress_arr = []

    # Start the optimization loop
    tbar = tqdm.tqdm(range(iters + 1))
    for i in tbar:
        optimizer.zero_grad()

        batch_ids = torch.arange(N)
        cur_wfe = wfe_batch[batch_ids]
        pred_1 = p1_model(plane_wave_1, cur_wfe)

        if args.loss_fn == 'L2':
            loss = (pred_1 - observations[batch_ids]).pow(2).sum() / loss_scale
        else:
            loss = (pred_1 - observations[batch_ids]).abs().sum() / loss_scale

        loss.backward()
        optimizer.step()
        scheduler.step()

        e_loss = loss.item() / len(cur_wfe)
        wfe_err = F.l1_loss(wfe_batch, wfe_gt)
        img_loss_arr.append(e_loss)
        wfe_err_arr.append(wfe_err.item())

        if i % vis_freq == 0 and i > 0:
            with torch.no_grad():
                e_pred = p1_model(plane_wave_1, wfe_batch).detach().cpu()
            est_residual = torch.relu(out - e_pred).detach().cpu().numpy()
            est_residual = est_residual / est_residual.max()
            est_residual = np.mean(est_residual, axis=0)
            est_residual = np.uint8(255 * cm.viridis(est_residual))
            imageio.imsave(f'{vis_dir}/vis_est_res_{i}.png', est_residual)

        if args.log_progress:
            cur_est = torch.relu(observations[batch_ids] - pred_1).detach().cpu().numpy()
            cur_est = cur_est / cur_est.max()
            cur_est = np.mean(cur_est, axis=0)
            cur_est = np.uint8(255 * cm.viridis(cur_est))
            progress_arr.append(cur_est)
        tbar_out = {'loss': e_loss, 'wfe_err': wfe_err.item()}
        tbar.set_postfix(tbar_out)

    if args.log_progress:
        progress_arr = np.stack(progress_arr)
        imageio.mimsave(f'{vis_dir}/progress.mp4', progress_arr[::50], 
                        'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 5, })

    with torch.no_grad():
        est_star_psf = p1_model(plane_wave_1, wfe_batch).detach().cpu().numpy()
    true_star_psf = out_1.numpy()
    est_residual_psf = torch.relu(out - est_star_psf).detach().cpu().numpy()
    true_residual_psf = (out - out_1).detach().cpu().numpy()

    psnr_init_star = peak_signal_noise_ratio(true_star_psf / true_star_psf.max(), init_est_star_psf / true_star_psf.max(), data_range=1)
    ssim_init_star = structural_similarity(true_star_psf / true_star_psf.max(), init_est_star_psf / true_star_psf.max(), data_range=1, channel_axis=0)
    psnr_star = peak_signal_noise_ratio(true_star_psf / true_star_psf.max(), est_star_psf / true_star_psf.max(), data_range=1)
    ssim_star = structural_similarity(true_star_psf / true_star_psf.max(), est_star_psf / true_star_psf.max(), data_range=1, channel_axis=0)

    div_norm = true_residual_psf.max()
    true_residual_psf = true_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm
    init_est_residual_psf = init_est_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm
    est_residual_psf = est_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm

    psnr_init_residual = peak_signal_noise_ratio(true_residual_psf, init_est_residual_psf, data_range=1)
    ssim_init_residual = structural_similarity(true_residual_psf, init_est_residual_psf, data_range=1, channel_axis=0)
    snr_init_residual = np.mean(init_est_residual_psf) / np.std(init_est_residual_psf)
    print(f'PSNR for the initial residual PSF: {psnr_init_residual:.2f} dB')
    print(f'SSIM for the initial residual PSF: {ssim_init_residual:.4f}')
    print(f'SNR for the initial residual PSF: {snr_init_residual:.2f}')

    psnr_residual = peak_signal_noise_ratio(true_residual_psf, est_residual_psf, data_range=1)
    ssim_residual = structural_similarity(true_residual_psf, est_residual_psf, data_range=1, channel_axis=0)
    snr_residual = np.mean(est_residual_psf) / np.std(est_residual_psf)
    print(f'PSNR for the residual PSF: {psnr_residual:.2f} dB')
    print(f'SSIM for the residual PSF: {ssim_residual:.4f}')
    print(f'SNR for the residual PSF: {snr_residual:.2f}')

    wfe_mse = F.mse_loss(wfe_batch / wfe_gt.max(), wfe_gt / wfe_gt.max()).item()
    wfe_psnr = np.log10(1 / wfe_mse)

    quality_arr = np.array([psnr_init_star, ssim_init_star, psnr_init_residual, ssim_init_residual,
                            psnr_star, ssim_star, psnr_residual, ssim_residual, wfe_mse, wfe_psnr])
    np.save(f'{vis_dir}/quality.npy', quality_arr)

    # Plot the loss curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(img_loss_arr)
    ax[0].set_title('Image Loss')
    ax[0].set_xlabel('Iterations')
    ax[1].plot(wfe_err_arr)
    ax[1].set_title('WFE Loss')
    ax[1].set_xlabel('Iterations')
    plt.savefig(f'{vis_dir}/losses.png')

    # Compare the GT WFEs and final WFEs
    vis_wfe_gt = wfe_gt.detach().cpu().numpy()
    vis_wfe_batch = wfe_batch.detach().cpu().numpy()
    vis_wfe = np.concatenate([vis_wfe_gt, vis_wfe_batch], axis=2)
    vis_wfe = (vis_wfe - vis_wfe.min()) / (vis_wfe.max() - vis_wfe.min())
    vis_wfe = np.uint8(cm.viridis(vis_wfe) * 255)
    ffmpeg_kargs = {'macro_block_size': None, 'ffmpeg_params': ['-s','512x256', '-v', '0'], 'fps': 4}
    imageio.mimsave(f'{vis_dir}/vis_wfe.mp4', vis_wfe, 'FFMPEG', **ffmpeg_kargs)
