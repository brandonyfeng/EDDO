from run_JWST_current_smaller_FOV_decenter_tryincnoise import PointPropagate, Wavefront, GridOffsetModule, AngleOffsetModule, FluxOffsetModule
import numpy as np 
import torch
import torch.nn.functional as F
from dl_utils import arcsec2rad
import matplotlib.pyplot as plt 
import torch.nn as nn
from run_JWST_current_smaller_FOV import load_data
from torchvision.transforms import v2
import os

DEVICE = 'cuda'

data_dir = './HR8799stuff/before_data'
num_wl = 21
oversample = 2
wf_npix = 1024

num_ints = 1
use_blur = False
compute_RMS_WFEs = False

reference_file =  'REF-HD220657_original/reference_00001.npy'

# Fit results are here:
folder = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/vis/HR8799_ref_NOTSHIFT_CENTERINGWHUUU/AA2/initial_fit_reference'

# Save FM here:
folnice = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HR8799stuff/FIT_FMs/angoff4717_truealmost3_nonircamoff'

if not os.path.isdir(folnice):
    os.mkdir(folnice)

includeangleoffset = False

custom_angle_offsets = np.array([4.7839155, 17.065592], dtype=np.float32)
print('angle offsets is ', custom_angle_offsets)
print('fold is ', folnice)
includenircamoffsets = False 
includeentranceopd = True 
includefluxcorrection = True


aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=data_dir, num_wl=num_wl, oversample=oversample)

aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]
nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
wlen_weights = torch.FloatTensor(wlen_weights)

sampledWFEs = np.load(f'{data_dir}/masks_2048/observation_opd.npy')
sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
wfe_batch = sampledWFEs.contiguous().to(DEVICE)

if num_ints > 1:
    sampledWFEs_after = np.load('./HR8799stuff/after_data/masks_2048/observation_opd.npy')
    sampledWFEs_after = np.flip(sampledWFEs_after, axis=0)[None]
    sampledWFEs_after = torch.from_numpy(sampledWFEs_after.copy()).float()
    sampledWFEs_after = F.interpolate(sampledWFEs_after[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch_after = sampledWFEs_after.contiguous().to(DEVICE)

if num_ints > 1:
    wfe_batch_list = [wfe_batch, wfe_batch_after]
else:
    wfe_batch_list = [wfe_batch]

contrast_normalization = 0.003716380479003919
sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data; VERY ROUGH! To be fixed
photons_normalization = 90578.00102527262 * sim_to_real_scaling *1e4 / 2251.24456327477#THREE CORRECTIONs # mJy/sr
peak_flux_star  = nn.Parameter(torch.FloatTensor([photons_normalization / contrast_normalization]))

star_offset_x = 0.4
star_offset_y = -1.2
psf_pixel_scale = 0.062424185
offset_STAR = nn.Parameter(torch.FloatTensor([star_offset_x * arcsec2rad(psf_pixel_scale), star_offset_y * arcsec2rad(psf_pixel_scale)]))

# Set up the propagation model parameters
shift = [0.0, 0.0]
pixel = True
focal_length = None
npixels = wf_npix
inverse = False
diameter = 6.603464

true_pixel_scale = diameter / npixels
psf_pixel_scale = arcsec2rad(psf_pixel_scale)
prop_args = (npixels, wlen_weights[0], true_pixel_scale, 80, psf_pixel_scale, focal_length, shift, pixel, inverse)

prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=2, num_det_px=80).to(DEVICE) for _ in range(num_ints)]

wavefronts_list1 = [Wavefront(wf_npix, diameter, wl, peak_flux_star, offset_STAR).to(DEVICE) for wl in wlen_weights[0]]



final_opd = np.load(folder+'/last_iteration_ENTRANCE_OPD_oversample_2_wl_sampling_21.npy')
final_nircam_opd = np.load(folder+'/last_iteration_NIRCAM_OPD_oversample_2_wl_sampling_21.npy')
angles_offset_res = np.load(folder+'/last_iteration_ANGLES_OFFSET_oversample_2_wl_sampling_21.npy')
final_opd = np.flip(final_opd, axis=0)

fluxcorrection_res = np.load(folder+'/last_iteration_FLUXCORRECTION_oversample_2_wl_sampling_21.npy')

if num_ints >1:
    final_opd_1 = np.load(folder+'/last_iteration_ENTRANCE_OPD_oversample_2_wl_sampling_21_1.npy')
    final_nircam_opd_1 = np.load(folder+'/last_iteration_NIRCAM_OPD_oversample_2_wl_sampling_21_1.npy')
    angles_offset_res_1 = np.load(folder+'/last_iteration_ANGLES_OFFSET_oversample_2_wl_sampling_21_1.npy')
    final_opd_1 = np.flip(final_opd_1, axis=0)


# final_nircam_opd = np.flip(final_nircam_opd, axis=0)
# final_nircam_opd_1 = np.flip(final_nircam_opd_1, axis=0)

sampledWFEs = np.flip(final_opd, axis=0)[None]
sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
wfe_batch = sampledWFEs.contiguous().to(DEVICE)

if num_ints > 1:
    sampledWFEs_after = np.flip(final_opd_1, axis=0)[None]
    sampledWFEs_after = torch.from_numpy(sampledWFEs_after.copy()).float()
    sampledWFEs_after = F.interpolate(sampledWFEs_after[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch_after = sampledWFEs_after.contiguous().to(DEVICE)


    wfe_batch_list = [wfe_batch, wfe_batch_after]

else:
    wfe_batch_list = [wfe_batch]

nra_mod_off = GridOffsetModule(1024,1024)
nra_mod_off.grid = nn.Parameter(torch.from_numpy(final_nircam_opd[None].copy()).to(DEVICE))
angle_obj = AngleOffsetModule()
angle_obj.data = nn.Parameter(torch.from_numpy(angles_offset_res.copy()).to(DEVICE))
fluxcorrection_res_obj = FluxOffsetModule()
fluxcorrection_res_obj.data = nn.Parameter(torch.from_numpy(fluxcorrection_res.copy()).to(DEVICE))




if includenircamoffsets:
    prop_models[0].nircam_offsets = nra_mod_off

if includeangleoffset:
    prop_models[0].angle_offsets = angle_obj
else:
    if custom_angle_offsets is not None:
        angle_obj.data = nn.Parameter(torch.from_numpy(custom_angle_offsets.copy()).to(DEVICE))
        prop_models[0].angle_offsets = angle_obj

if includefluxcorrection:
    prop_models[0].flux_correction = fluxcorrection_res_obj

if num_ints > 1:
    nra_mod_off_1 = GridOffsetModule(1024,1024)
    nra_mod_off_1.grid = nn.Parameter(torch.from_numpy(final_nircam_opd_1[None].copy()).to(DEVICE))
    angle_obj_1 = AngleOffsetModule()
    angle_obj_1.data = nn.Parameter(torch.from_numpy(angles_offset_res_1.copy()).to(DEVICE))
    prop_models[1].nircam_offsets = nra_mod_off_1
    prop_models[1].angle_offsets = angle_obj_1



with torch.no_grad():
    pred = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
    pred = torch.mean(torch.cat(pred, 0), 0)


if use_blur:
    krn = 31
    sgma = 25
    with torch.no_grad():
        wfe_batch_list_blurred = [v2.GaussianBlur(kernel_size=krn, sigma=sgma)(wfe_batch_list[0][None]),v2.GaussianBlur(kernel_size=krn, sigma=sgma)(wfe_batch_list[1][None])]
        pred_blur_opd = [prop_models[j](wavefronts_list1, wfe_batch_list_blurred[j][0], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred_blur_opd = torch.mean(torch.cat(pred_blur_opd, 0), 0)

else:
    pred_blur_opd = pred


if compute_RMS_WFEs:
    def get_rms(arr, mask):
        return torch.sqrt((arr[(mask != 0)] ** 2).mean())

    transmission = np.load('/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/4050_data_for_diff_modeling/group_2_before/masks_1024/primary_transmission_1024.npy')
    transmission = np.flip(transmission, axis=0)[None]
    transmission = torch.from_numpy(transmission.copy()).float()
    transmission = F.interpolate(transmission[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    transmission = transmission.contiguous().to(DEVICE)

    with torch.no_grad():
        # print('batch shpe is ', wfe_batch_list[0].shape)
        # print('batch none shpe is ', wfe_batch_list[0][None].shape)
        # print('transmissino shape is ', transmission.shape)
        # print('transmission none shape is ', transmission[None].shape)
        rms_orig_0 = get_rms(wfe_batch_list[0], transmission)
        rms_orig_1 = get_rms(wfe_batch_list[1], transmission)
        wfe_batch_list_blurred = [v2.GaussianBlur(kernel_size=krn, sigma=sgma)(wfe_batch_list[0][None]),v2.GaussianBlur(kernel_size=krn, sigma=sgma)(wfe_batch_list[1][None])]
        
        rms_orig_0_blur = get_rms(wfe_batch_list_blurred[0], transmission[None])
        rms_orig_1_blur = get_rms(wfe_batch_list_blurred[1], transmission[None])

        wfe_batch_list_blurred_scaled = [wfe_batch_list_blurred[0]/rms_orig_0_blur *rms_orig_0, wfe_batch_list_blurred[1]/rms_orig_1_blur *rms_orig_1]
        pred_blur_opd_scaledrms = [prop_models[j](wavefronts_list1, wfe_batch_list_blurred_scaled[j][0], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred_blur_opd_scaledrms = torch.mean(torch.cat(pred_blur_opd_scaledrms, 0), 0)


np.save(folnice+'/model_asfit.npy',pred.detach().cpu().numpy())
#np.save(folnice+'/model_opdblurred.npy',pred_blur_opd.detach().cpu().numpy())

#np.save(folnice+'/model_opdblurred_scalerms.npy',pred_blur_opd_scaledrms.detach().cpu().numpy())

#np.save(folnice+'/blur_opd.npy',wfe_batch_list_blurred[0].detach().cpu().numpy())
#np.save(folnice+'/blur_opd_1.npy',wfe_batch_list_blurred[1].detach().cpu().numpy())

#np.save(folnice+'/blur_opd_scalerms.npy',wfe_batch_list_blurred_scaled[0].detach().cpu().numpy())
#np.save(folnice+'/blur_opd_1_scalerms.npy',wfe_batch_list_blurred_scaled[1].detach().cpu().numpy())

plt.figure()
h =plt.imshow(pred.detach().cpu().squeeze(), origin='lower')
plt.colorbar(h)
plt.savefig(folnice+'/yes_1.png')

edge1, edge2 = 160 - int(40), 160 + int(40)

# NOW WE DO NOT HAVE AN IMAGE CENTERED AT THE ARRAY, so use known coronagraph position to crop at the right pixels
shift_y,shift_x = -14, 10

# ref1_data_NOTcentered_CROPPED = ref1_data_NOTcentered[(120-sfhitx):(200-sfhitx),(120-sfhity):(200-sfhity)]

real_im = np.load(f'{data_dir}/real_data/{reference_file}')[0, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
real_im = real_im.astype(np.float32)
np.save(folnice+'/reference_data.npy',real_im)