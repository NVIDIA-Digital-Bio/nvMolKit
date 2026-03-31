import torch
import triton
import triton.language as tl

TILE_X = 32
TILE_Y = 32
# TODO: L2 Cache Optimizations
@triton.jit
def _popcount32(x):
    # SWAR bit count fallback for Triton builds without tl.popcount.
    x = x.to(tl.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x * 0x01010101
    return (x >> 24).to(tl.int32)


def _check_fp_tensor(name: str, x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype int32")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")


def _check_vec_tensor(
    name: str,
    x: torch.Tensor,
    expected_len: int,
    *,
    allow_larger: bool = False,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype int32")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    if allow_larger:
        if x.numel() < expected_len:
            raise ValueError(
                f"{name} must have length >= {expected_len}, got {x.numel()}"
            )
    else:
        if x.numel() != expected_len:
            raise ValueError(f"{name} must have length {expected_len}, got {x.numel()}")


@triton.jit
def _similarity_neighbor_kernel(
    x_ptr,
    y_ptr,
    neighbors_ptr,
    n,
    m,
    K,
    x_stride_n,
    x_stride_k,
    y_stride_n,
    y_stride_k,
    threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < n
    mask_n = offs_n < m

    norm_x = tl.zeros((BLOCK_M,), dtype=tl.int32)
    norm_y = tl.zeros((BLOCK_N,), dtype=tl.int32)
    dots = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_offset + kk
            k_mask = k_idx < K
            xk = tl.load(
                x_ptr + offs_m * x_stride_n + k_idx * x_stride_k,
                mask=mask_m & k_mask,
                other=0,
            )
            yk = tl.load(
                y_ptr + offs_n * y_stride_n + k_idx * y_stride_k,
                mask=mask_n & k_mask,
                other=0,
            )
            norm_x += _popcount32(xk)
            norm_y += _popcount32(yk)
            dots += _popcount32(xk[:, None] & yk[None, :])

    denom = norm_x[:, None] + norm_y[None, :] - dots
    valid = mask_m[:, None] & mask_n[None, :] & (denom > 0)
    similarity = tl.where(valid, dots.to(tl.float32) / denom.to(tl.float32), 0.0)
    is_neighbor = valid & (similarity >= threshold)

    row_counts = tl.sum(is_neighbor.to(tl.int32), axis=1)
    tl.atomic_add(neighbors_ptr + offs_m, row_counts, mask=mask_m)


@triton.jit
def _subtract_similarity_neighbor_kernel(
    x_ptr,
    y_ptr,
    neighbors_ptr,
    n,
    m,
    K,
    x_stride_n,
    x_stride_k,
    y_stride_n,
    y_stride_k,
    threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < n
    mask_n = offs_n < m

    norm_x = tl.zeros((BLOCK_M,), dtype=tl.int32)
    norm_y = tl.zeros((BLOCK_N,), dtype=tl.int32)
    dots = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_offset + kk
            k_mask = k_idx < K
            xk = tl.load(
                x_ptr + offs_m * x_stride_n + k_idx * x_stride_k,
                mask=mask_m & k_mask,
                other=0,
            )
            yk = tl.load(
                y_ptr + offs_n * y_stride_n + k_idx * y_stride_k,
                mask=mask_n & k_mask,
                other=0,
            )
            norm_x += _popcount32(xk)
            norm_y += _popcount32(yk)
            dots += _popcount32(xk[:, None] & yk[None, :])

    denom = norm_x[:, None] + norm_y[None, :] - dots
    valid = mask_m[:, None] & mask_n[None, :] & (denom > 0)
    similarity = tl.where(valid, dots.to(tl.float32) / denom.to(tl.float32), 0.0)
    is_neighbor = valid & (similarity >= threshold)

    row_counts = -tl.sum(is_neighbor.to(tl.int32), axis=1)
    tl.atomic_add(neighbors_ptr + offs_m, row_counts, mask=mask_m) 


@triton.jit
def _remove_largest_cluster_kernel(
    x_ptr,
    center_id,
    is_free_ptr,
    neighbors_ptr,
    cluster_count_ptr,
    cluster_indices_ptr,
    threshold,
    indices_ptr,
    n,
    K,
    x_stride_n,
    x_stride_k,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_mask = row < n

    pa = tl.zeros((), dtype=tl.int32)
    pb = tl.zeros((), dtype=tl.int32)
    dot = tl.zeros((), dtype=tl.int32)
    
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_offset + kk
            k_mask = k_idx < K
            center_k = tl.load(x_ptr + center_id * x_stride_n + k_idx * x_stride_k, mask=k_mask, other=0)
            row_k = tl.load(
                x_ptr + row * x_stride_n + k_idx * x_stride_k,
                mask=row_mask & k_mask,
                other=0,
            )
            pa += _popcount32(center_k)
            pb += _popcount32(row_k)
            dot += _popcount32(row_k & center_k)

    union = pa + pb - dot
    row_is_free = tl.load(is_free_ptr + row, mask=row_mask, other=0)
    valid = row_mask & (row_is_free != 0) & (union > 0)
    similarity = tl.where(valid, dot.to(tl.float32) / union.to(tl.float32), 0.0)
    is_neighbor = valid & (similarity >= threshold)

    orig_idx = tl.load(indices_ptr + row, mask=row_mask, other=0)
    neighbor_slot = tl.atomic_add(cluster_count_ptr + 0, 1, mask=is_neighbor)
    tl.store(cluster_indices_ptr + neighbor_slot, orig_idx, mask=is_neighbor)
    tl.store(is_free_ptr + row, 0, mask=is_neighbor)

    degree = tl.load(neighbors_ptr + row, mask=row_mask, other=0)
    is_singleton = row_mask & (~is_neighbor) & (degree == 1)
    singleton_slot = tl.atomic_add(cluster_count_ptr + 1, -1, mask=is_singleton)
    tl.store(cluster_indices_ptr + singleton_slot, orig_idx, mask=is_singleton)
    tl.store(is_free_ptr + row, 0, mask=is_singleton)


def similarity_neighbor(
    x: torch.Tensor,
    y: torch.Tensor,
    neighbors: torch.Tensor,
    threshold: float,
) -> None:
    _check_fp_tensor("x", x)
    _check_fp_tensor("y", y)
    _check_vec_tensor("neighbors", neighbors, x.shape[0])
    if x.device != y.device or x.device != neighbors.device:
        raise ValueError("x, y, and neighbors must be on the same CUDA device")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")

    n = x.shape[0]
    m = y.shape[0]
    K = x.shape[1]
    grid = (triton.cdiv(n, TILE_X), triton.cdiv(m, TILE_Y))
    _similarity_neighbor_kernel[grid](
        x,
        y,
        neighbors,
        n,
        m,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        float(threshold),
        BLOCK_M=TILE_X,
        BLOCK_N=TILE_Y,
        BLOCK_K=32,
        num_warps=8,
    )


def subtract_similarity_neighbor(
    x: torch.Tensor,
    y: torch.Tensor,
    neighbors: torch.Tensor,
    threshold: float,
) -> None:
    _check_fp_tensor("x", x)
    _check_fp_tensor("y", y)
    _check_vec_tensor("neighbors", neighbors, x.shape[0])
    if x.device != y.device or x.device != neighbors.device:
        raise ValueError("x, y, and neighbors must be on the same CUDA device")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")

    n = x.shape[0]
    m = y.shape[0]
    K = x.shape[1]
    grid = (triton.cdiv(n, TILE_X), triton.cdiv(m, TILE_Y))
    _subtract_similarity_neighbor_kernel[grid](
        x,
        y,
        neighbors,
        n,
        m,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        float(threshold),
        BLOCK_M=TILE_X,
        BLOCK_N=TILE_Y,
        BLOCK_K=32,
        num_warps=8,
    )


def remove_largest_cluster(
    x: torch.Tensor,
    id: int,
    is_free: torch.Tensor,
    neighbors: torch.Tensor,
    cluster_count: torch.Tensor,
    cluster_indices: torch.Tensor,
    threshold: float,
    indices: torch.Tensor,
) -> None:
    _check_fp_tensor("x", x)
    n = x.shape[0]
    K = x.shape[1]
    _check_vec_tensor("is_free", is_free, n)
    _check_vec_tensor("neighbors", neighbors, n)
    _check_vec_tensor("cluster_indices", cluster_indices, n, allow_larger=True)
    _check_vec_tensor("indices", indices, n)
    _check_vec_tensor("cluster_count", cluster_count, 2)
    if not (0 <= id < n):
        raise ValueError(f"id must be in [0, {n}), got {id}")
    if (
        x.device != is_free.device
        or x.device != neighbors.device
        or x.device != cluster_count.device
        or x.device != cluster_indices.device
        or x.device != indices.device
    ):
        raise ValueError("all tensors must be on the same CUDA device")

    grid = (n,)
    _remove_largest_cluster_kernel[grid](
        x,
        id,
        is_free,
        neighbors,
        cluster_count,
        cluster_indices,
        float(threshold),
        indices,
        n,
        K,
        x.stride(0),
        x.stride(1),
        BLOCK_K=32,
        num_warps=1,
    )
