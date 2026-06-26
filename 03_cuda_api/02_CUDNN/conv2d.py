import torch
import torch.nn.functional as F

# Define the same dimensions as in the CUDA code
width = 4
height = 4
kernel_size = 3
in_channels = 1
out_channels = 1
batch_size = 1

# Define the input tensor
input_values = torch.tensor(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ],
    dtype=torch.float32,
).reshape(batch_size, in_channels, height, width)

# Define the kernel tensor
kernel_values = torch.tensor(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ],
    dtype=torch.float32,
).reshape(out_channels, in_channels, kernel_size, kernel_size)

# Perform the convolution
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = F.conv2d(input_values, kernel_values, padding=kernel_size // 2)
end.record()
# Wait for the events to be recorded
torch.cuda.synchronize()
# Calculate the elapsed time in milliseconds
elapsed_time_ms = start.elapsed_time(end)
print(f"Elapsed time for convolution: {elapsed_time_ms:.6f} ms")

# Print the input, kernel, and output tensors
print("Input:")
print(input_values)
print("\nKernel:")
print(kernel_values)
print("\nOutput:")
print(output)

# Print the output in a flattened format for easier comparison
print("\nFlattened output:")
print(output.flatten().tolist())
print(len(output.flatten().tolist()))

