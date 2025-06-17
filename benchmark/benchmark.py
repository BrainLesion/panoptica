import timeit
from typing import Tuple, Union

import cc3d
import numpy as np

# scipy needs to be installed to run this benchmark, we use cc3d as it is quicker for 3D data
from scipy import ndimage

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. GPU benchmarks will be skipped.")


def generate_random_binary_mask(size: Tuple[int, int, Union[int, None]]) -> np.ndarray:
    """
    Generate a random binary mask with the specified size.

    Args:
        size (Tuple[int, int, Union[int, None]]): Size of the mask. Use (height, width) for 2D or (height, width, depth) for 3D.

    Returns:
        np.ndarray: Random binary mask.
    """
    if len(size) == 2:
        return np.random.randint(0, 2, (*size, 1), dtype=bool)
    elif len(size) == 3:
        return np.random.randint(0, 2, size, dtype=bool)
    else:
        raise ValueError(
            "Invalid dimension for volume size. Use (height, width) for 2D or (height, width, depth) for 3D."
        )


def benchmark_scipy(mask: np.ndarray):
    """
    Benchmark the performance of scipy.ndimage.label for connected component labeling.

    Args:
        mask (np.ndarray): Binary mask to label.

    Returns:
        float: Time taken to label the mask in seconds.
    """

    def label_scipy():
        ndimage.label(mask)

    scipy_time = timeit.timeit(label_scipy, number=10)
    return scipy_time


def benchmark_cc3d(mask: np.ndarray):
    """
    Benchmark the performance of cc3d.connected_components for connected component labeling.

    Args:
        mask (np.ndarray): Binary mask to label.

    Returns:
        float: Time taken to label the mask in seconds.
    """

    def label_cc3d():
        cc3d.connected_components(mask, return_N=True)

    cc3d_time = timeit.timeit(label_cc3d, number=10)
    return cc3d_time


def benchmark_cupy(mask: np.ndarray):
    """
    Benchmark the performance of cupy.ndimage.label for connected component labeling on GPU.

    Args:
        mask (np.ndarray): Binary mask to label.

    Returns:
        float: Time taken to label the mask in seconds, or None if CuPy is not available.
    """
    if not CUPY_AVAILABLE:
        return None

    # Transfer data to GPU
    mask_gpu = cp.asarray(mask)

    # Warmup phase
    for _ in range(3):
        cp_ndimage.label(mask_gpu)
        cp.cuda.Stream.null.synchronize()

    def label_cupy():
        cp_ndimage.label(mask_gpu)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete

    cupy_time = timeit.timeit(label_cupy, number=10)

    # Clean up GPU memory
    del mask_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return cupy_time


def run_benchmarks(volume_sizes: Tuple[Tuple[int, int, Union[int, None]]]) -> None:
    """
    Run benchmark tests for connected component labeling with different volume sizes.

    Args:
        volume_sizes (Tuple[Tuple[int, int, Union[int, None]]]): List of volume sizes to test.
            Use (height, width) for 2D or (height, width, depth) for 3D.

    Returns:
        None
    """
    for size in volume_sizes:
        mask = generate_random_binary_mask(size)

        scipy_time = benchmark_scipy(mask)
        cc3d_time = benchmark_cc3d(mask)
        cupy_time = benchmark_cupy(mask)

        print(f"Volume Size: {size}")
        print(f"Scipy Time: {scipy_time:.4f} seconds")
        print(f"CC3D Time: {cc3d_time:.4f} seconds")
        if cupy_time is not None:
            print(f"CuPy Time: {cupy_time:.4f} seconds")
        else:
            print("CuPy Time: Not available")
        print()


if __name__ == "__main__":
    # Define a list of volume sizes to test
    volume_sizes = [
        (500, 500),  # 2D volume
        (1000, 1000),  # 2D volume
        (2000, 2000),  # 2D volume
        (50, 50, 50),  # 3D volume
        (100, 100, 100),  # 3D volume
        (200, 200, 200),  # 3D volume
        (512, 512, 512),  # 3D volume
    ]

    run_benchmarks(volume_sizes)
