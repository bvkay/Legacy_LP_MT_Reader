import matplotlib.pyplot as plt
import numpy as np
import os

"""
'orange box' LP MT Data Reader
------------------------------
-      BK @ UoA, AuScope     -
-         [2025-02-20]       -
-           v0.0.2           -
------------------------------
"""

def process_binary_files(site_name, output_file_path, sample_number, displayvalue2, average=False):
    """
    Reads and processes all binary data files for a given site based on the MAGALL2.C structure.

    Args:
        site_name (str): Base name of the binary files (e.g., "HFM4-").
        output_file_path (str): Path to the output ASCII file.
        sample_number (int): Total number of samples to process.
        displayvalue2 (int): Interval for saving/displaying data.
        average (bool): Whether to average values over intervals.
    """
    try:
        # Open the output file
        with open(output_file_path, "w") as outfile:
            # Initialize counters and arrays
            NCHANNELS = 8
            counts = [0] * NCHANNELS
            avgcounts = [0] * NCHANNELS

            print(f"Processing binary files for site: {site_name}")

            # Iterate through all binary files matching the site name pattern
            file_index = 0
            while True:
                binary_file_path = f"{site_name}{file_index:03d}.BIN"

                if not os.path.exists(binary_file_path):
                    print(f"No more files found after: {binary_file_path}")
                    break

                print(f"Processing file: {binary_file_path}")

                # Open and process the binary file
                with open(binary_file_path, "rb") as bin_file:
                    # Read the first line: Sample Rate (Hexadecimal)
                    first_line = bin_file.readline().decode("ascii", errors="ignore").strip()
                    sample_rate = int(first_line, 16)  # Convert from Hexadecimal to Decimal
                    print(f"Sample Rate: {sample_rate} (Decimal)")

                    # Read the second line: Date and Time Stamp (ASCII)
                    second_line = bin_file.readline().decode("ascii", errors="ignore").strip()
                    print(f"Date and Time: {second_line}")

                    # Read the next 4 characters for the Filter Point (Hexadecimal)
                    filter_point_bytes = bin_file.read(4).decode("ascii", errors="ignore").strip()
                    filter_point = int(filter_point_bytes, 16)  # Convert from Hexadecimal to Decimal
                    print(f"Filter Point: {filter_point} (Decimal), {filter_point:X}h (Hexadecimal)")

                    # Calculate Samples per Second
                    samples_per_second = 10_000_000 / (512 * filter_point)
                    print(f"Samples per Second: {samples_per_second:.6f}")

                    # Read and process samples
                    displayvalue = int(samples_per_second * displayvalue2)
                    samplecount = 0

                    while samplecount < sample_number:
                        # Process channels 0-2 (3 bytes each)
                        for i in range(3):
                            tempchar1 = bin_file.read(1)
                            tempchar2 = bin_file.read(1)
                            tempchar3 = bin_file.read(1)
                            if len(tempchar1) != 1 or len(tempchar2) != 1 or len(tempchar3) != 1:
                                raise ValueError("Incomplete data for 3-byte channel")
                            counts[i] = (int.from_bytes(tempchar1, byteorder="big", signed=False) << 16) + \
                                        (int.from_bytes(tempchar2, byteorder="big", signed=False) << 8) + \
                                        int.from_bytes(tempchar3, byteorder="big", signed=False)

                        # Process channels 3-4 (2 bytes each)
                        for i in range(3, 5):
                            tempchar1 = bin_file.read(1)
                            tempchar2 = bin_file.read(1)
                            if len(tempchar1) != 1 or len(tempchar2) != 1:
                                raise ValueError("Incomplete data for 2-byte channel")
                            counts[i] = (int.from_bytes(tempchar1, byteorder="big", signed=False) << 8) + \
                                        int.from_bytes(tempchar2, byteorder="big", signed=False)

                        # Process channel 5 (1 byte)
                        tempchar1 = bin_file.read(1)
                        if len(tempchar1) != 1:
                            raise ValueError("Incomplete data for 1-byte channel")
                        counts[5] = int.from_bytes(tempchar1, byteorder="big", signed=False)

                        # Process channels 6-7 (3 bytes each)
                        for i in range(6, 8):
                            tempchar1 = bin_file.read(1)
                            tempchar2 = bin_file.read(1)
                            tempchar3 = bin_file.read(1)
                            if len(tempchar1) != 1 or len(tempchar2) != 1 or len(tempchar3) != 1:
                                raise ValueError("Incomplete data for 3-byte channel")
                            counts[i] = (int.from_bytes(tempchar1, byteorder="big", signed=False) << 16) + \
                                        (int.from_bytes(tempchar2, byteorder="big", signed=False) << 8) + \
                                        int.from_bytes(tempchar3, byteorder="big", signed=False)

                        # Read and discard the extra byte after the 8 channels
                        extra_byte = bin_file.read(1)
                        if len(extra_byte) != 1:
                            raise ValueError("Missing extra byte after 8 channels")

                        if average:
                            # Handle averaging if enabled
                            for i in range(NCHANNELS):
                                avgcounts[i] += counts[i]  # Accumulate counts for averaging

                            if (samplecount + 1) % displayvalue == 0:
                                # Calculate averages
                                avgcounts = [avg // displayvalue for avg in avgcounts]
                                # Write averaged data to file in structured format
                                outfile.write("  ".join(f"{value:08d}" for value in avgcounts) + "\n")
                                # Reset averages
                                avgcounts = [0] * NCHANNELS
                        else:
                            # Write raw data to file in structured format
                            outfile.write("  ".join(f"{value:08d}" for value in counts) + "\n")

                        # Increment sample count
                        samplecount += 1

                # Move to the next file
                file_index += 1

            print(f"Processing complete for site: {site_name}")
    except Exception as e:
        print(f"Error processing binary files for site {site_name}: {e}")

def plot_channels(output_file_path):
    """
    Plots the first 5 channels from the processed output file as subplots in one figure.

    Args:
        output_file_path (str): Path to the output ASCII file containing processed data.
    """
    try:
        # Load data from the output file
        data = np.loadtxt(output_file_path, dtype=int)

        # Extract the first 5 channels
        channels = data[:, :5].T  # Transpose for easier plotting

        # Plot the channels
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(channels[i], label=f"Channel {i}")
            ax.set_ylabel(f"CH{i}")
            ax.legend()
        axes[-1].set_xlabel("Sample Number")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting channels: {e}")

# Example usage
site_name = "TEST/HFM4-"  # Replace with the base name of your binary files
output_file_path = "output9.txt"  # Replace with the desired output file path
sample_number = 36000  # Replace with the actual sample count
displayvalue2 = 1  # Interval for saving/displaying data
process_binary_files(site_name, output_file_path, sample_number, displayvalue2, average=True)
plot_channels(output_file_path)
