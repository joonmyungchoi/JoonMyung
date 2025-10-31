from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torch

def rangeBlock(block, vmin=0, vmax=5):
    loss = torch.arange(vmin, vmax, (vmax - vmin) / block, requires_grad=False).unsqueeze(dim=1)
    return loss

def columnRename(df, ns):
    for n in ns:
        if n[0] in df.columns:
            df.rename(columns = {n[0]: n[1]}, inplace = True)
#     columnRemove(df, ['c1', 'c2' ... ])


def columnRemove(df, ns):
    delList = []
    for n in ns:
        if n in df.columns:
            delList.append(n)
    df.drop(delList, axis=1, inplace=True)
#     columnRename(df, [['c1_p', 'c1_a'] , ['c2_p', 'c2_a']])


def normalization(t, type = 0):
    if type == 0:
        return t / t.max()
    elif type == 1:
        return t / t.min()


def getTransform(train = False, totensor = False, resize=True):

    if not resize:
        transform = lambda x: x
    else:
        transform = []

        transform.append(transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC)) \
            if train else transform.append(transforms.Resize((224, 224), interpolation=3))

        if totensor:
            transform.append(transforms.ToTensor())
            transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform)

    return transform


import torch
import torch.nn.functional as F

def _gaussian_kernel2d(ky: int, kx: int, base_truncate: float = 3.0,
                       dtype=torch.float32, device=None) -> torch.Tensor:
    """
    (ky, kx) 크기의 합=1 가우시안 커널 생성.
    sigma는 ksize와 base_truncate로 역산: radius = truncate * sigma
      => sigma = radius / truncate
    base_truncate는 고정 상수(모양만 결정), 사용자는 건드릴 필요 없음.
    """
    assert ky % 2 == 1 and kx % 2 == 1, "kernel_size는 홀수여야 합니다."
    ry, rx = (ky - 1) // 2, (kx - 1) // 2
    # 너무 뾰족/평평해지는 걸 막기 위한 최소치
    sig_y = max(ry / max(base_truncate, 1e-6), 1e-6)
    sig_x = max(rx / max(base_truncate, 1e-6), 1e-6)

    y = torch.arange(-ry, ry + 1, dtype=dtype, device=device)
    x = torch.arange(-rx, rx + 1, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    k = torch.exp(-0.5 * ((yy / sig_y) ** 2 + (xx / sig_x) ** 2))
    k = k / k.sum()
    return k  # (ky, kx)

def _delta_kernel2d(ky: int, kx: int, dtype=torch.float32, device=None) -> torch.Tensor:
    k = torch.zeros((ky, kx), dtype=dtype, device=device)
    k[ky // 2, kx // 2] = 1.0
    return k

def smooth_kernel(
    heatmap: torch.Tensor,         # (B, H, W)
    kernel_size: list = [9, 9],    # 홀수
    s: float = 0.0,                # 0이면 원본, ↓할수록 가우시안 쪽으로
    norm: int = 1.0,
    pad_mode: str = "reflect",     # 'reflect' | 'replicate' | 'constant'
) -> torch.Tensor:
    dtype, device = heatmap.dtype, heatmap.device
    kernel_size_y, kernel_size_x = kernel_size
    # 커널들 생성
    G = _gaussian_kernel2d(kernel_size_y, kernel_size_x, dtype=dtype, device=device)
    D = _delta_kernel2d(kernel_size_y, kernel_size_x, dtype=dtype, device=device)

    # 강도 s: 0이면 원본, 1이면 순수 가우시안. 1을 넘겨도 동작하도록 1-포화 매핑.
    K = (1.0 - s) * D + s * G  # 합=1 유지

    py, px = kernel_size_y // 2, kernel_size_x // 2
    K = K.view(1, 1, kernel_size_y, kernel_size_x)

    x = heatmap.unsqueeze(1)  # (B,1,H,W)
    x = F.pad(x, (px, px, py, py), mode=pad_mode)
    y = F.conv2d(x, K)        # (B,1,H,W)

    if norm:
        y = (y - y.min()) / (y.max() - y.min())
    return y.squeeze(1)