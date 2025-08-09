import torch

def parallel_cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    A parallel implementation of the cumulative sum (prefix sum) algorithm.
    This is designed to run efficiently on a GPU using vectorized PyTorch operations,
    avoiding Python loops and CPU-based computations. It is an implementation of the
    Hillis-Steele algorithm.

    Args:
        x (torch.Tensor): A 1D tensor of numbers.

    Returns:
        torch.Tensor: A tensor of the same shape as x, where each element is the
                      cumulative sum of x up to that point.
    """
    # For stability and correctness, operate on a cloned tensor
    y = x.clone()
    n = y.shape[0]
    stride = 1

    # The loop runs log2(n) times, making it highly efficient for large n.
    while stride < n:
        # Create a shifted version of the tensor, padded with zeros at the beginning.
        # This represents the values from the previous elements in the sequence.
        shifted = torch.nn.functional.pad(y[:-stride], (stride, 0))
        
        # Add the shifted tensor to the original. This is a fully vectorized,
        # out-of-place operation that will execute on the GPU.
        y = y + shifted
        
        # Double the stride for the next pass.
        stride *= 2
        
    return y