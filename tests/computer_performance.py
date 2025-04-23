"""
This Program evaluates the performance of the PC
on PyTorch Tensor multiplication.

---------- Scores ----------
Omar's laptop:          650
Omar's old desktop:     210

"""
import torch
import time
import matplotlib.pyplot as plt


def measure_performance_with_plot(device='cuda', max_time=1.0, size_increment=200):
    # Set device to CUDA if available and requested
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU instead.")

    # Initialize lists to store sizes and times
    tensor_sizes = []
    execution_times = []

    # Start with an initial tensor size and incrementally increase it
    current_size = size_increment

    # Enable interactive mode for matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], marker='o', linestyle='-', color='b')  # Empty line object
    ax.set_title(f'Matrix Multiplication Performance (Real-Time) on {device}')
    ax.set_xlabel('Tensor Size (N x N)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.axhline(max_time, color='r', linestyle='--', label=f'Max Time ({max_time} s)')  # Horizontal max_time line
    ax.grid(True)
    ax.legend()

    while True:
        # Create random tensors on the selected device
        tensor_a = torch.rand((current_size, current_size), device=device)
        tensor_b = torch.rand((current_size, current_size), device=device)

        # Measure matrix multiplication performance
        start_time = time.time()

        # Perform matrix multiplication
        _ = torch.matmul(tensor_a, tensor_b)

        # Synchronize in case of GPU to ensure accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time

        # Store the results
        tensor_sizes.append(current_size)
        execution_times.append(elapsed_time)

        print(f"{current_size:6} {elapsed_time:6.3f} s")

        # Update the plot in real-time
        line.set_xdata(tensor_sizes)  # Update the x data (tensor sizes)
        line.set_ydata(execution_times)  # Update the y data (execution times)

        # Adjust plot limits dynamically
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Autoscale the view

        plt.draw()  # Redraw the plot
        plt.pause(0.1)  # Pause to allow the plot to update

        # Check if we've exceeded the maximum allowed time
        if elapsed_time > max_time:
            print(f"\nTime limit of {max_time} seconds reached. Stopping.")

            # Calculate the intersection point using the last two points
            if len(tensor_sizes) >= 2:
                x1, x2 = tensor_sizes[-2], tensor_sizes[-1]
                y1, y2 = execution_times[-2], execution_times[-1]

                # Equation of the line: y = m*x + b
                m = (y2 - y1) / (x2 - x1)  # Slope
                b = y1 - m * x1  # Intercept

                # Find x where y = max_time
                max_n = (max_time - b) / m

                # Find final score
                final_score = 2 * max_n**3 / max_time

                print(f"\n\033[32m FINAL SCORE: {final_score*10**(-9):.0f} GFLOPS\033[0m")

            break

        # Increase the tensor size for the next iteration
        current_size += size_increment

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Run performance measurement by default on GPU (if possible) with a max time of 2 seconds
    measure_performance_with_plot()
