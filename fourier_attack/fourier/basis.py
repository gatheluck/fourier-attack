import torch
import torch.fft as fft  # this is needed to pass mypy check


def spectrum_to_basis(spectrum: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert spectrum matrix to Fourier basis by 2D FFT.
    Shape of returned basis is (B, H, H). NOTE: If H!=W, returned basis might be wrong.
    In order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args
        spectrum: Shape should be (B,H,W//2+1).
        normalize: If True, basis is l2 normalized.
    """
    _, H, _ = spectrum.size()
    basis = fft.irfftn(spectrum, s=(H, H), dim=(-2, -1))

    if normalize:
        return basis / basis.norm(dim=(-2, -1))[:, None, None]
    else:
        return basis


def get_basis_spectrum(height: int, width: int) -> torch.Tensor:
    """
    Get all specrum matrics of 2D Fourier basis.
    Shape of return is (height*width, height, width)
    """
    x = torch.arange(height * width)
    return torch.nn.functional.one_hot(x).view(-1, height, width).float()
