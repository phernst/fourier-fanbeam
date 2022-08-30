"""
Implementation of "A New Fourier Method for Fan Beam Reconstruction",
Shuang-Ren Zhao and Horst Balling, 1995 IEEE Nuclear Science Symposium and
Medical Imaging Conference Record, 1995, pp. 1287-1291 vol.2,
doi: 10.1109/NSSMIC.1995.510494.
"""
import sys
import timeit
from typing import Optional

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from torch_radon import FanBeam


def _calculate_capital_pi_map(gamma_max: float, gamma_map: torch.Tensor):
    capital_pi_map = torch.zeros_like(gamma_map)
    capital_pi_map[(-gamma_max <= gamma_map-2*np.pi) & (gamma_map-2*np.pi < gamma_max)] = 1
    capital_pi_map[(-gamma_max <= gamma_map) & (gamma_map < gamma_max)] = 1
    capital_pi_map[(-gamma_max <= gamma_map+2*np.pi) & (gamma_map+2*np.pi < gamma_max)] = 1
    return capital_pi_map


def _compute_gamma_max(source_distance: float, det_distance: float, det_count: int, det_spacing: float):
    return np.arctan2(det_count/2*det_spacing, source_distance + det_distance)


def _compute_theta_map(size: int, det_spacing: float):
    fftfreq = torch.fft.fftfreq(size, det_spacing, device='cuda')
    omega_j, omega_i = torch.meshgrid(fftfreq*2*np.pi, -fftfreq*2*np.pi, indexing='ij')
    theta_map = torch.zeros_like(omega_i)
    theta_map[omega_i > 0] = torch.atan(omega_j[omega_i > 0]/omega_i[omega_i > 0])
    theta_map[omega_i < 0] = torch.atan(omega_j[omega_i < 0]/omega_i[omega_i < 0]) + np.pi
    theta_map[(omega_i == 0) & (omega_j > 0)] = np.pi/2
    theta_map[(omega_i == 0) & (omega_j < 0)] = -np.pi/2
    return theta_map


def _compute_eta_map(size: int, det_spacing: float):
    fftfreq = torch.fft.fftfreq(size, det_spacing)
    omega_j, omega_i = torch.meshgrid(fftfreq*2*np.pi, -fftfreq*2*np.pi, indexing='ij')
    return torch.sqrt(omega_i**2 + omega_j**2)


def _compute_low_pass_map(eta_max: float, eta_map: torch.Tensor):
    low_pass_map = torch.ones_like(eta_map)
    low_pass_map[eta_map > eta_max] = 0
    return low_pass_map


def _compute_hann_map(eta_max: float, eta_map: torch.Tensor):
    max_freq = int(eta_map.shape[0]*eta_max)
    hann_x, hann_y = torch.meshgrid(
        torch.hann_window(max_freq),
        torch.hann_window(max_freq),
        indexing='xy',
    )
    hann_map = hann_x*hann_y

    out_size = eta_map.shape[0]
    pad_front = (out_size - max_freq)//2
    pad_back = out_size - (max_freq + pad_front)
    hann_map = F.pad(hann_map, (pad_front, pad_back)*2)

    return torch.fft.fftshift(hann_map)


def test_hann_map():
    hann_map = _compute_hann_map(.5, torch.zeros(512, 512))
    plt.imshow(hann_map.cpu().numpy())
    plt.show()


# projections: [k, s]
# gamma_map: [k, i, j]
# returns [k, i, j]
def _rebin_fanbeam_projections(projections: torch.Tensor, gamma_map: torch.Tensor, gamma_max: float):
    projections = projections.unsqueeze(1).unsqueeze(1)  # [k, 1, 1, s]
    grid = torch.zeros(
        gamma_map.shape[0],
        gamma_map.shape[1],
        gamma_map.shape[2],
        2,
        dtype=torch.float,
        device=projections.device)
    grid[..., 0] = -gamma_map.tan()/np.tan(gamma_max)
    grid[..., 1] = 0
    return F.grid_sample(projections, grid, align_corners=False, mode='bilinear')[:, 0]


# k dimension first
# theta_map: [i, j]
# beta_k = [k]
# returns gamma(k, i, j)
def _compute_gamma_map(theta_map: torch.Tensor, beta_k: torch.Tensor):
    gamma_map = theta_map[..., None] - beta_k
    return gamma_map.permute(2, 0, 1)


# calculates the Fourier transform of the object function
def _compute_full_fourier_space(fanbeam_rebin: torch.Tensor,
                                pi_map: torch.Tensor,
                                source_distance: float,
                                gamma_map: torch.Tensor,
                                delta_beta: float,
                                const_exp: torch.Tensor):
    exp_map = torch.exp(gamma_map.sin()*const_exp)
    elements = pi_map*fanbeam_rebin*exp_map*gamma_map.cos()
    return elements.sum(-3)*delta_beta*source_distance


# reconstructs fanbeam projections
def reconstruct(projections: torch.Tensor, beta_k: torch.Tensor,
                radon: FanBeam, output_size: Optional[int] = None,
                apodization: Optional[str] = None, max_frequency: float = 1.0,
                oversampling: Optional[int] = None,
                shortscan: bool = False) -> torch.Tensor:
    assert len(beta_k) > 1

    det_spacing = radon.projection.cfg.det_spacing_u
    src_distance = radon.projection.cfg.s_dist
    det_distance = radon.projection.cfg.d_dist
    det_count = radon.projection.cfg.det_count_u

    output_size = output_size if output_size is not None else projections.shape[-1]
    theta_map = _compute_theta_map(output_size, det_spacing)
    gamma_max = _compute_gamma_max(src_distance, det_distance, det_count, det_spacing)
    oversampling = oversampling \
        if oversampling is not None \
        else _calculate_oversampling(gamma_max, output_size, len(beta_k))
    eta_map = _compute_eta_map(output_size, det_spacing).cuda()
    low_pass_fn = _compute_hann_map if apodization == 'hann' else _compute_low_pass_map
    low_pass_map = low_pass_fn(max_frequency if max_frequency is not None else 1., eta_map).cuda()

    full_fourier_space = torch.zeros_like(theta_map, dtype=torch.complex64)
    beta_diff = beta_k[1] - beta_k[0]
    const_exp = -1j*src_distance*eta_map*det_spacing

    def _single_reco(dbeta: float):
        gamma_map = _compute_gamma_map(theta_map, beta_k + dbeta)  # [k, i, j]
        pi_map = _calculate_capital_pi_map(gamma_max, gamma_map)  # [k, i, j]
        img = nib.Nifti1Image(np.fft.fftshift(gamma_map.cpu().numpy().transpose(), axes=(0, 1)), np.eye(4))
        nib.save(img, 'bla.nii.gz')
        fanbeam_rebin = _rebin_fanbeam_projections(projections[0], gamma_map, gamma_max)
        return _compute_full_fourier_space(
            fanbeam_rebin,
            pi_map,
            src_distance,
            gamma_map,
            ((beta_k[-1] - beta_k[0] + beta_diff) if shortscan else 2*np.pi)/len(beta_k),
            const_exp,
        )

    if oversampling == 0:
        full_fourier_space = _single_reco(0.)
    else:
        # nearest neighbor if oversampling
        first_half = (k*beta_diff*2**-oversampling for k in range(2**oversampling//2))
        second_half = (
            k*beta_diff*2**-oversampling
            for k in range(2**oversampling//2, 2**oversampling)
        )

        for dbeta in first_half:
            full_fourier_space += _single_reco(dbeta)
        projections = torch.roll(projections, -1, -2)
        for dbeta in second_half:
            full_fourier_space += _single_reco(dbeta)

    full_fourier_space *= low_pass_map
    full_fourier_space /= 2**oversampling
    if shortscan:
        full_fourier_space = full_fourier_space[:, :full_fourier_space.shape[-1]//2 + 1]
        return torch.fft.ifftshift(torch.fft.irfft2(full_fourier_space))
    return torch.fft.ifftshift(torch.fft.ifft2(full_fourier_space)).real


# nn-upsampling for an equiangularly undersampled fanbeam sinogram
# angle between first and last projection must equal angle between other projections
# this is an alternative for the oversampling in `reconstruct`
def custom_upsample(under_sino: torch.Tensor, upsampling: int) -> torch.Tensor:
    if upsampling == 1:
        return under_sino

    full_views = under_sino.shape[2]*upsampling
    first_up = F.interpolate(
        under_sino,
        scale_factor=((full_views + 1 - upsampling)/((full_views + 1)//upsampling), 1.),
        mode='nearest',
        recompute_scale_factor=True,
    )
    last_part = torch.cat((
        under_sino[..., -1:, :],
        under_sino[..., :1, :]
        ), dim=-2)
    last_up = F.interpolate(
        last_part,
        scale_factor=((2 + upsampling - 1)/2, 1.),
        mode='nearest',
        recompute_scale_factor=True,
    )[..., 1:-1, :]
    return torch.cat((first_up, last_up), dim=-2)


# calculates the oversampling that is necessary to avoid Fourier ringing
# returns exponent of power of two as needed for `reconstruct`
def _calculate_oversampling(gamma_max: float, fourier_size: int, num_projections: int) -> int:
    return int(np.ceil(np.log2(np.pi*fourier_size/gamma_max/num_projections)))


def test_reco_time(path_to_volume: str):
    volume = nib.load(path_to_volume).get_fdata()
    np_central_slice = volume[..., volume.shape[-1]//2].transpose()[::2, ::2]
    central_slice = torch.from_numpy(np_central_slice).float().cuda()[None]

    beta_k = torch.deg2rad(torch.linspace(0, 360, 811, device='cuda'))[:-1]
    src_distance: float = 720.
    det_distance: float = 1080.
    det_count: int = 256
    det_spacing: float = 3.645161
    fanradon = FanBeam(
        angles=beta_k,
        src_dist=src_distance,
        det_dist=det_distance,
        det_count=det_count,
        det_spacing=det_spacing,
        volume=central_slice[0].shape[0],
    )
    projections = fanradon.forward(central_slice)
    t = timeit.Timer(
        lambda: reconstruct(
            projections, beta_k, fanradon,
            output_size=central_slice.shape[-1],
            apodization='hann',
            oversampling=0,
        ),
        globals=globals())
    times = t.repeat(250, 1)
    print(np.min(times), np.median(times))


def test_reco(path_to_volume: str):
    volume = nib.load(path_to_volume).get_fdata()
    np_central_slice = volume[..., volume.shape[-1]//2].transpose()[::2, ::2]
    central_slice = torch.from_numpy(np_central_slice).float().cuda()[None]
    plt.imshow(central_slice[0].cpu().numpy(), vmin=0)

    beta_k = torch.deg2rad(torch.linspace(-15, 180+15, 811, device='cuda'))[:-1]
    source_distance: float = 720.
    det_distance: float = 1080.
    det_count: int = 255
    det_spacing: float = 3.645161
    fanradon = FanBeam(
        angles=beta_k,
        src_dist=source_distance,
        det_dist=det_distance,
        det_count=det_count,
        det_spacing=det_spacing,
        volume=central_slice[0].shape[0],
    )
    sino = fanradon.forward(central_slice)
    plt.figure()
    plt.imshow(sino[0].cpu().numpy())

    reco = reconstruct(
        sino, beta_k, fanradon,
        output_size=central_slice.shape[-1],
        apodization='hann',
        shortscan=True,
    )
    plt.figure()
    plt.imshow(reco.cpu().numpy(), vmin=0, vmax=central_slice.max())

    full_fourier_space = torch.fft.fft2(reco)
    plt.figure()
    plt.imshow(np.fft.fftshift((full_fourier_space.abs()+1).log().cpu().numpy()))

    power_gt = np.log(np.abs(np.fft.fft2(central_slice[0].cpu().numpy())) + 1)
    plt.figure()
    plt.imshow(np.fft.fftshift(power_gt))

    fbp_reco = fanradon.backprojection(fanradon.filter_sinogram(sino, filter_name='hann'))
    plt.figure()
    plt.imshow(fbp_reco[0].cpu().numpy(), vmin=0, vmax=central_slice.max())
    plt.show()


if __name__ == '__main__':
    test_reco(sys.argv[1])
    # test_reco_time(sys.argv[1])
