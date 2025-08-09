import torch
from phase import parallel_cumsum, parallel_one_pole_lowpass

def test_parallel_cumsum():
    """
    Tests the gradient of the parallel_cumsum function using gradcheck.
    """
    print("\n--- Testing Gradient for parallel_cumsum ---")
    # gradcheck requires double-precision floating point numbers
    # and a requires_grad=True flag on the input.
    # We test on the target device using float32, as float64 is not supported by MPS.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Running test on CPU.")

    # Use a small tensor for the test to keep it fast.
    test_input = torch.randn(16, dtype=torch.float32, device=device, requires_grad=True)
    
    # The function to be tested must return a single tensor.
    # Our function already does this.
    try:
        # We use float32, so we may need a slightly higher tolerance.
        is_correct = torch.autograd.gradcheck(parallel_cumsum, test_input, eps=1e-3, atol=1e-3)
        if is_correct:
            print("Gradient for parallel_cumsum is CORRECT.")
        else:
            print("!!! Gradient for parallel_cumsum is INCORRECT. !!!")
    except Exception as e:
        print(f"!!! An error occurred during gradcheck for parallel_cumsum: {e} !!!")


def test_parallel_one_pole_lowpass():
    """
    Tests the gradient of the parallel_one_pole_lowpass function.
    """
    print("\n--- Testing Gradient for parallel_one_pole_lowpass ---")
    # We test on the target device using float32, as float64 is not supported by MPS.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Running test on CPU.")

    # The function takes multiple arguments, so we package them as a tuple.
    # We will check the gradient with respect to the input signal `x` and the
    # time-varying `cutoff_freq`.
    sample_rate = 48000
    x = torch.randn(16, dtype=torch.float32, device=device, requires_grad=True)
    cutoff_freq = torch.full((16,), 1000.0, dtype=torch.float32, device=device, requires_grad=True)
    
    # We need to wrap the function in a lambda to match the format gradcheck expects,
    # as it can only check functions that take a tuple of tensors as input.
    func = lambda x, cutoff: parallel_one_pole_lowpass(x, cutoff, sample_rate)

    try:
        # We use float32, so we may need a slightly higher tolerance.
        is_correct = torch.autograd.gradcheck(func, (x, cutoff_freq), eps=1e-3, atol=1e-3)
        if is_correct:
            print("Gradient for parallel_one_pole_lowpass is CORRECT.")
        else:
            print("!!! Gradient for parallel_one_pole_lowpass is INCORRECT. !!!")
    except Exception as e:
        print(f"!!! An error occurred during gradcheck for parallel_one_pole_lowpass: {e} !!!")


if __name__ == '__main__':
    print("Running gradient checks for custom synthesis functions...")
    test_parallel_cumsum()
    test_parallel_one_pole_lowpass()
    print("\n--- Gradient checks complete. ---")
