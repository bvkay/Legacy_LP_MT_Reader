import os
import io
import argparse
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.signal import welch, coherence  # Removed savgol_filter
from Legacy_LP_MT_Reader import BinaryReader  # Updated binary reader module

LOG_FILE = "process_binary.log"


def write_log(message, level="INFO"):
    """Writes a log message to the log file.
    
    Args:
        message (str): The log message.
        level (str, optional): Log level. Defaults to "INFO".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {level} - {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)


def dict_to_datetime(time_dict):
    """Converts a time dictionary to a datetime object.
    
    Args:
        time_dict (dict): Dictionary with keys "year", "month", "day", "hour", "minute", "second".
    
    Returns:
        datetime.datetime: Constructed datetime.
    """
    return datetime.datetime(
        year=time_dict["year"],
        month=time_dict["month"],
        day=time_dict["day"],
        hour=time_dict["hour"],
        minute=time_dict["minute"],
        second=time_dict["second"]
    )


def apply_drift_correction_to_df(df, metadata):
    """Applies linear drift correction to the DataFrame.
    
    Correction: corrected_time = time + (time / total_duration) * time_drift,
    and adds a datetime column.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column (seconds from survey start).
        metadata (dict): Contains "start_time", "finish_time", and "time_drift".
    
    Returns:
        pd.DataFrame: DataFrame with "time_corrected" and "time_corrected_dt".
    """
    start_dt = dict_to_datetime(metadata["start_time"])
    finish_dt = dict_to_datetime(metadata["finish_time"])
    total_duration = (finish_dt - start_dt).total_seconds()
    drift_value = metadata["time_drift"]

    df["time_corrected"] = df["time"] + (df["time"] / total_duration) * drift_value
    df["time_corrected_dt"] = df["time_corrected"].apply(
        lambda s: start_dt + datetime.timedelta(seconds=s)
    )
    return df


def convert_counts_to_physical_units(df, metadata):
    """Converts raw counts into physical units.
    
    Magnetics:
      Bx = (chan0/2**23 - 1) * 70000  
      Bz = (chan1/2**23 - 1) * 70000  
      By = -(chan2/2**23 - 1) * 70000  
    Electrics:
      Ex = -(chan7/2**23 - 1) * (100000 / xarm)  
      Ey = -(chan6/2**23 - 1) * (100000 / yarm)
    
    Args:
        df (pd.DataFrame): DataFrame with channels "chan0", "chan1", etc.
        metadata (dict): Contains "xarm" and "yarm".
    
    Returns:
        pd.DataFrame: DataFrame with new columns "Bx", "Bz", "By", "Ex", and "Ey".
    """
    divisor = 2**23
    df["Bx"] = (df["chan0"].astype(float) / divisor - 1.0) * 70000.0
    df["Bz"] = (df["chan1"].astype(float) / divisor - 1.0) * 70000.0
    df["By"] = -(df["chan2"].astype(float) / divisor - 1.0) * 70000.0
    xarm = metadata.get("xarm", 1.0)
    yarm = metadata.get("yarm", 1.0)
    df["Ex"] = -(df["chan7"].astype(float) / divisor - 1.0) * (100000.0 / xarm)
    df["Ey"] = -(df["chan6"].astype(float) / divisor - 1.0) * (100000.0 / yarm)
    return df


def rotate_data(df, metadata):
    """Rotates horizontal magnetic (and optionally electric) fields.
    
    Computes angle_avg = mean(arctan2(By, Bx)) and rotates the horizontal fields.
    If metadata["erotate"] == 1, the electric fields are also rotated.
    
    Args:
        df (pd.DataFrame): DataFrame with "Bx", "By", "Bz", "Ex", "Ey".
        metadata (dict): Contains "erotate".
    
    Returns:
        pd.DataFrame: DataFrame with rotated fields ("Hx", "Dx", "Z_rot", and optionally "Ex_rot", "Ey_rot").
    """
    angles = np.arctan2(df["By"].values, df["Bx"].values)
    angle_avg = np.mean(angles)
    write_log(f"Computed rotation angle: {np.degrees(angle_avg):.2f} degrees")

    df["Hx"] = df["Bx"] * np.cos(angle_avg) + df["By"] * np.sin(angle_avg)
    df["Dx"] = df["By"] * np.cos(angle_avg) - df["Bx"] * np.sin(angle_avg)
    df["Z_rot"] = df["Bz"]

    if metadata.get("erotate", 0) == 1:
        df["Ex_rot"] = df["Ex"] * np.cos(angle_avg) + df["Ey"] * np.sin(angle_avg)
        df["Ey_rot"] = df["Ey"] * np.cos(angle_avg) - df["Ex"] * np.sin(angle_avg)
    else:
        df["Ex_rot"] = df["Ex"]
        df["Ey_rot"] = df["Ey"]
    return df


### Smoothing Functions: Only Median/MAD and Adaptive Median Filtering are retained ###

def smooth_outlier_points(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], window=50, threshold=3.0):
    """Applies outlier detection and smoothing using rolling median and MAD.
    
    Outliers are detected and then interpolated linearly.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        channels (list, optional): Channels to process. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        tuple: (df, outlier_info) where outlier_info maps channels to outlier intervals.
    """
    outlier_info = {}
    for ch in channels:
        mask = detect_outliers(df[ch], window=window, threshold=threshold)
        intervals = get_intervals_from_mask(mask, df["time"])
        outlier_info[ch] = intervals
        df[ch] = smooth_outliers(df[ch], mask)
    return df, outlier_info


def detect_outliers(series, window=50, threshold=3.0):
    """Detects outliers using a rolling median and MAD approach.
    
    Args:
        series (pd.Series): Input time-series.
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        pd.Series: Boolean Series marking outliers.
    """
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    abs_diff = (series - rolling_median).abs()
    rolling_mad = abs_diff.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = rolling_mad.replace(0, 1e-6)
    return abs_diff > threshold * rolling_mad


def smooth_outliers(series, outlier_mask):
    """Interpolates linearly over regions marked as outliers.
    
    Args:
        series (pd.Series): Input time-series.
        outlier_mask (pd.Series): Boolean mask indicating outliers.
    
    Returns:
        pd.Series: Smoothed series.
    """
    series_clean = series.copy()
    series_clean[outlier_mask] = np.nan
    return series_clean.interpolate(method="linear")


def get_intervals_from_mask(mask, time_values):
    """Extracts (start_time, end_time) intervals where the mask is True.
    
    Args:
        mask (pd.Series): Boolean mask from outlier detection.
        time_values (pd.Series): Corresponding time values.
    
    Returns:
        list: List of tuples (start_time, end_time).
    """
    intervals = []
    in_interval = False
    start_time = None
    for i, flag in enumerate(mask):
        if flag and not in_interval:
            in_interval = True
            start_time = time_values.iloc[i]
        elif not flag and in_interval:
            intervals.append((start_time, time_values.iloc[i - 1]))
            in_interval = False
    if in_interval:
        intervals.append((start_time, time_values.iloc[-1]))
    return intervals


def smooth_median_mad(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], window=50, threshold=3.0):
    """Wrapper for the median/MAD smoothing method.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        channels (list, optional): Channels to smooth. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        tuple: (df, outlier_info)
    """
    return smooth_outlier_points(df, channels=channels, window=window, threshold=threshold)


def smooth_adaptive_median(signal, min_window=51, max_window=2500, threshold=10.0):
    """Applies an adaptive median filter to a 1D numpy array.
    
    For each sample, starts with a window of size min_window (odd) and expands until the sample
    is within threshold * MAD of the median or until max_window is reached.
    
    Args:
        signal (np.ndarray): Input 1D array.
        min_window (int, optional): Minimum window size (odd). Defaults to 51.
        max_window (int, optional): Maximum window size (odd). Defaults to 2500.
        threshold (float, optional): Threshold multiplier for MAD. Defaults to 10.0.
    
    Returns:
        np.ndarray: Smoothed array.
    """
    smoothed = np.copy(signal)
    n = len(signal)
    if min_window % 2 == 0:
        min_window += 1
    if max_window % 2 == 0:
        max_window += 1

    for i in range(n):
        window_size = min_window
        smoothed_value = signal[i]
        while window_size <= max_window:
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window = signal[start:end]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            if mad == 0:
                mad = 1e-6
            if np.abs(signal[i] - med) <= threshold * mad:
                smoothed_value = med
                break
            window_size += 2
        smoothed[i] = smoothed_value
    return smoothed


def smooth_adaptive(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], min_window=3, max_window=2500, threshold=10.0):
    """Applies adaptive median filtering to specified channels of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column and measurement channels.
        channels (list, optional): Channels to smooth. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        min_window (int, optional): Minimum window size (odd). Defaults to 3.
        max_window (int, optional): Maximum window size (odd). Defaults to 2500.
        threshold (float, optional): Threshold multiplier for MAD. Defaults to 10.0.
    
    Returns:
        tuple: (df_adapt, {})
    """
    df_adapt = df.copy()
    for ch in channels:
        data = df_adapt[ch].values
        df_adapt[ch] = smooth_adaptive_median(data, min_window=min_window, max_window=max_window, threshold=threshold)
    return df_adapt, {}


### Plotting Functions ###

def plot_power_spectra(df, channels, fs, nperseg=1024, save_plots=False, site_name="UnknownSite"):
    """Plots the power spectral density (PSD) for specified channels.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        channels (list): List of channels to plot.
        fs (float): Sampling frequency.
        nperseg (int, optional): Samples per segment for Welch's method. Defaults to 1024.
        save_plots (bool, optional): If True, saves the figure; otherwise displays it.
        site_name (str, optional): Site name for the saved filename.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4), sharey=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        f, Pxx = welch(df[ch].values, fs=fs, nperseg=nperseg)
        ax.semilogy(f, Pxx)
        ax.set_title(f'PSD of {ch}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.grid(True)
    plt.tight_layout()
    if save_plots:
        filename = f"{site_name}_power_spectra.png"
        plt.savefig(filename)
        plt.close(fig)
        write_log(f"Power spectra plot saved as {filename}")
    else:
        plt.show()


def plot_coherence_plots(df, pairs, fs, nperseg=1024, save_plots=False, site_name="UnknownSite"):
    """Plots coherence for specified channel pairs.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        pairs (list): List of tuples (e.g., [("Bx", "Ey"), ...]).
        fs (float): Sampling frequency.
        nperseg (int, optional): Samples per segment for coherence calculation. Defaults to 1024.
        save_plots (bool, optional): If True, saves the figure; otherwise displays it.
        site_name (str, optional): Site name for the saved filename.
    
    Returns:
        None
    """
    num_pairs = len(pairs)
    ncols = 2
    nrows = (num_pairs + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4 * nrows))
    axes = axes.flatten()
    for ax, (ch1, ch2) in zip(axes, pairs):
        f, Cxy = coherence(df[ch1].values, df[ch2].values, fs=fs, nperseg=nperseg)
        ax.plot(f, Cxy)
        ax.set_title(f'Coherence: {ch1} vs. {ch2}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.grid(True)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)
    plt.tight_layout()
    if save_plots:
        filename = f"{site_name}_coherence.png"
        plt.savefig(filename)
        plt.close(fig)
        write_log(f"Coherence plot saved as {filename}")
    else:
        plt.show()


def plot_physical_channels(df, boundaries=None, plot_boundaries=True,
                           smoothed_intervals=None, plot_smoothed_windows=True,
                           tilt_corrected=False, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots physical channel data in subplots.
    
    Since tilt correction overwrites the original 'Bx' and 'By' columns,
    this function always uses "Bx", "By", etc.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        boundaries (list): List of time values for file boundaries.
        plot_boundaries (bool): Whether to draw vertical lines at boundaries.
        smoothed_intervals (dict): Dictionary of smoothed window intervals.
        plot_smoothed_windows (bool): Whether to shade smoothed windows.
        tilt_corrected (bool): For labeling purposes (unused in column mapping).
        save_plots (bool): If True, saves the figure; otherwise displays it.
        site_name (str): Site name used for naming the saved file.
        output_dir (str, optional): Directory to save plots. Defaults to current directory.
    
    Returns:
        None
    """
    channel_mapping = {"Bx": "Bx", "By": "By", "Bz": "Bz", "Ex": "Ex", "Ey": "Ey"}

    fig, axes = plt.subplots(len(channel_mapping), 1, figsize=(10, 15), sharex=True)
    for ax, (label, col) in zip(axes, channel_mapping.items()):
        ax.plot(df["time"], df[col], label="Uncorrected")
        ax.plot(df["time_corrected"], df[col], label="Drift Corrected", linestyle="--")
        mean_val = df[col].mean()
        ax.axhline(mean_val, color='black', linestyle=':', label=f"Mean = {mean_val:.2f}")
        ax.set_title(label)
        ax.legend()
        ax.set_ylabel("Physical Units")
        if plot_boundaries and boundaries is not None:
            for b in boundaries:
                ax.axvline(x=b, color='green', linestyle='--', linewidth=1)
        if plot_smoothed_windows and smoothed_intervals is not None and label in smoothed_intervals:
            for interval in smoothed_intervals[label]:
                ax.axvspan(interval[0], interval[1], color='gray', alpha=0.3)
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_plots:
        filename = os.path.join(output_dir, f"{site_name}_physical_channels.png")
        plt.savefig(filename)
        plt.close(fig)
        write_log(f"Physical channels plot saved as {filename}")
    else:
        plt.show()


class ProcessBinary:
    """Processes binary MT files for a given site.
    
    Reads metadata, loads binary files using the instrument number (from param.mt),
    concatenates raw data, applies corrections (drift, rotation, tilt), optional smoothing,
    generates plots (or saves them), and saves raw and processed outputs.
    Output files and plots are named using the site name (derived from the input directory).
    """
    
    def __init__(self, input_dir, param_file, average=False, perform_freq_analysis=False,
                 plot_data=False, apply_smoothing=False, smoothing_window=2500, threshold_factor=10.0,
                 plot_boundaries=True, plot_smoothed_windows=True, plot_coherence=False,
                 log_first_rows=False, smoothing_method="median", sens_start=0, sens_end=5000,
                 skip_minutes=0):
        """
        Args:
            input_dir (str): Directory containing binary files and param.mt.
            param_file (str): Name of the parameter file.
            average (bool): Whether to average values over intervals.
            perform_freq_analysis (bool): Whether to perform frequency analysis.
            plot_data (bool): Whether to display physical channel plots.
            apply_smoothing (bool): Whether to apply smoothing.
            smoothing_window (int): Window size for smoothing.
            threshold_factor (float): Threshold multiplier for outlier detection.
            plot_boundaries (bool): Whether to plot file boundaries.
            plot_smoothed_windows (bool): Whether to shade smoothed windows.
            plot_coherence (bool): Whether to plot power spectra and coherence.
            log_first_rows (bool): Whether to log the first 5 rows from each file.
            smoothing_method (str): Smoothing method ("median" or "adaptive").
            sens_start (int): Start sample index for sensitivity test (unused now).
            sens_end (int): End sample index for sensitivity test (unused now).
            skip_minutes (int): Number of minutes to skip at the beginning of data.
        """
        self.input_dir = input_dir
        self.param_file = param_file
        self.average = average
        self.perform_freq_analysis = perform_freq_analysis
        self.plot_data = plot_data
        self.apply_smoothing = apply_smoothing
        self.smoothing_window = smoothing_window
        self.threshold_factor = threshold_factor
        self.plot_boundaries = plot_boundaries
        self.plot_smoothed_windows = plot_smoothed_windows
        self.plot_coherence = plot_coherence
        self.log_first_rows = log_first_rows
        self.smoothing_method = smoothing_method
        self.sens_start = sens_start
        self.sens_end = sens_end
        self.skip_minutes = skip_minutes
        self.metadata = self.read_metadata()
        # Use the folder (site) name for naming outputs.
        self.site_name = os.path.basename(os.path.normpath(self.input_dir))
        self.rotate = False       # Set via command-line.
        self.tilt_correction = False  # Set via command-line.
        self.save_plots = False   # Set via command-line.

    def read_metadata(self):
        """Reads metadata from the parameter file in the input directory.
        
        Returns:
            dict: Metadata dictionary.
        """
        metadata = {}
        try:
            param_file_path = os.path.join(self.input_dir, self.param_file)
            with open(param_file_path, "r") as file:
                lines = [line.split("!")[0].strip() for line in file.readlines() if line.strip()]
            metadata = {
                "instrument_number": lines[0],  # Used for loading binary files.
                "output_file": lines[1],
                "nstart": int(lines[2].split()[0]),
                "nend": int(lines[2].split()[1]),
                "xarm": float(lines[3].split()[0]),
                "yarm": float(lines[3].split()[1]),
                "erotate": int(lines[4]),
                "start_time": self._parse_time(lines[5]),
                "alarm_time": self._parse_time(lines[6]),
                "finish_time": self._parse_time(lines[7]),
                "sample_interval": int(lines[8]),
                "time_drift": int(lines[9]),
            }
            write_log("Metadata successfully loaded from " + param_file_path)
        except Exception as e:
            write_log(f"Error reading metadata file {self.param_file} from {self.input_dir}: {e}", level="ERROR")
        return metadata

    def _parse_time(self, time_line):
        """Parses a time string from the metadata file.
        
        Args:
            time_line (str): String containing time components.
        
        Returns:
            dict: Dictionary with keys "day", "month", "year", "hour", "minute", "second".
        """
        parts = list(map(int, time_line.split()))
        return {"day": parts[0], "month": parts[1], "year": parts[2],
                "hour": parts[3], "minute": parts[4], "second": parts[5]}

    def process_all_files(self):
        """Processes all binary files in the input directory.
        
        Loads binary files using the instrument number from metadata,
        concatenates raw data, applies corrections (drift, rotation, tilt),
        optional smoothing, generates plots (or saves them), and saves outputs.
        
        Returns:
            None
        """
        try:
            all_dfs = []
            file_boundaries = []
            cumulative_count = 0
            file_index = 0

            # Load binary files.
            while True:
                instrument = self.metadata.get("instrument_number", self.site_name)
                binary_file_path = os.path.join(self.input_dir, f"{instrument}-{file_index:03d}.BIN")
                if not os.path.exists(binary_file_path):
                    write_log(f"No more files found after: {binary_file_path}")
                    break
                dummy_outfile = io.StringIO()
                data_reader = BinaryReader(binary_file_path, self.metadata, self.average, log_first_rows=self.log_first_rows)
                df = data_reader.read_file(dummy_outfile)
                if df is not None:
                    all_dfs.append(df)
                    cumulative_count += df.shape[0]
                    file_boundaries.append(cumulative_count)
                file_index += 1

            if not all_dfs:
                write_log("No binary files found to process.", level="ERROR")
                return

            # Concatenate raw data.
            combined_raw_df = pd.concat(all_dfs, ignore_index=True)
            write_log("All files concatenated successfully (raw data).")

            # Save raw data.
            raw_columns = [col for col in combined_raw_df.columns if col.startswith("chan")]
            raw_output_path = os.path.join(self.input_dir, f"{self.site_name}_output_raw.txt")
            combined_raw_df[raw_columns].to_csv(raw_output_path, index=False, sep="\t")
            write_log(f"Raw data saved to {raw_output_path}")

            # Build continuous time column.
            sample_interval = self.metadata["sample_interval"]
            combined_raw_df["time"] = np.arange(len(combined_raw_df)) * sample_interval
            write_log("Continuous time column built using sample interval.")

            # Skip first X minutes if requested.
            if self.skip_minutes > 0:
                skip_seconds = self.skip_minutes * 60
                combined_raw_df = combined_raw_df[combined_raw_df["time"] >= skip_seconds].reset_index(drop=True)
                write_log(f"Skipped first {self.skip_minutes} minutes ({skip_seconds} seconds) of data.")

            # Apply drift correction.
            processed_df = apply_drift_correction_to_df(combined_raw_df.copy(), self.metadata)
            write_log("Drift correction applied to the dataset.")

            # Convert counts to physical units.
            processed_df = convert_counts_to_physical_units(processed_df, self.metadata)
            write_log("Counts converted to physical units.")

            # Optionally apply rotation.
            if self.rotate and self.metadata.get("erotate", 0) == 1:
                processed_df = rotate_data(processed_df, self.metadata)
                write_log("Data rotation applied.")

            # Optionally apply tilt correction.
            if self.tilt_correction:
                processed_df = tilt_correction(processed_df)
                write_log("Tilt correction applied.")

            # Smoothing section: Use either median or adaptive method.
            if self.apply_smoothing:
                if self.smoothing_method == "median":
                    processed_df, smoothed_intervals = smooth_median_mad(processed_df,
                                                                        channels=["Bx", "By", "Bz", "Ex", "Ey"],
                                                                        window=self.smoothing_window,
                                                                        threshold=self.threshold_factor)
                elif self.smoothing_method == "adaptive":
                    processed_df, _ = smooth_adaptive(processed_df,
                                                      channels=["Bx", "By", "Bz", "Ex", "Ey"],
                                                      min_window=3,
                                                      max_window=self.smoothing_window,
                                                      threshold=self.threshold_factor)
                    smoothed_intervals = {}
                else:
                    write_log("Unknown smoothing method selected.", level="ERROR")
                    return
            else:
                smoothed_intervals = None

            boundaries_in_time = ([b * sample_interval for b in file_boundaries[:-1]]
                                  if len(file_boundaries) > 1 else None)

            # Frequency analysis.
            if self.perform_freq_analysis:
                self.frequency_analysis(processed_df, "Combined Data")

            # Plot power spectra and coherence.
            if self.plot_coherence:
                fs = 1.0 / np.median(np.diff(processed_df["time"].values))
                plot_power_spectra(processed_df,
                                   channels=["Bx", "By", "Bz", "Ex", "Ey"],
                                   fs=fs,
                                   nperseg=1024,
                                   save_plots=self.save_plots,
                                   site_name=self.site_name)
                pairs = [("Bx", "Ey"), ("By", "Ex"), ("Bx", "By"), ("Ex", "Ey")]
                plot_coherence_plots(processed_df,
                                     pairs=pairs,
                                     fs=fs,
                                     nperseg=1024,
                                     save_plots=self.save_plots,
                                     site_name=self.site_name)
            # Plot physical channels.
            if self.plot_data:
                plot_physical_channels(processed_df,
                                       boundaries=boundaries_in_time,
                                       plot_boundaries=self.plot_boundaries,
                                       smoothed_intervals=smoothed_intervals,
                                       plot_smoothed_windows=self.plot_smoothed_windows,
                                       tilt_corrected=self.tilt_correction,
                                       save_plots=self.save_plots,
                                       site_name=self.site_name)
            # Save processed (scaled) output with 5 decimal places.
            scaled_columns = ["Bx", "By", "Bz", "Ex", "Ey"]
            scaled_output_path = os.path.join(self.input_dir, f"{self.site_name}_output_scaled.txt")
            processed_df[scaled_columns].to_csv(scaled_output_path, index=False, sep="\t", float_format="%.5f")
            write_log(f"Processed scaled data saved to {scaled_output_path}")
        except Exception as e:
            write_log(f"Error processing files: {e}", level="ERROR")

    def frequency_analysis(self, df, identifier):
        """Performs FFT-based frequency analysis if a 'data' column exists.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze.
            identifier (str): Label for the analysis run.
        """
        try:
            from scipy.fft import fft, fftfreq
            if 'data' in df.columns:
                data_values = df['data'].values
                N = len(data_values)
                T = self.metadata["sample_interval"]
                yf = fft(data_values)
                xf = fftfreq(N, T)[:N // 2]
                write_log(f"Frequency analysis on {identifier} completed.")
        except Exception as e:
            write_log(f"Error in frequency analysis for {identifier}: {e}", level="ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MT binary files for a given site.")
    parser.add_argument("--input_dir", required=True, help="Directory containing binary files and param.mt")
    parser.add_argument("--param_file", default="param.mt", help="Parameter file name (located in input_dir)")
    parser.add_argument("--plot_data", action="store_true", help="Plot physical channel data")
    parser.add_argument("--apply_smoothing", action="store_true", help="Apply outlier smoothing")
    parser.add_argument("--plot_boundaries", action="store_true", help="Plot file boundaries")
    parser.add_argument("--plot_smoothed_windows", action="store_true", help="Plot smoothed windows")
    parser.add_argument("--plot_coherence", action="store_true", help="Plot power spectra and coherence")
    parser.add_argument("--smoothing_window", type=int, default=2500, help="Window size for smoothing")
    parser.add_argument("--threshold_factor", type=float, default=10.0, help="Threshold multiplier for outlier detection")
    parser.add_argument("--perform_freq_analysis", action="store_true", help="Perform frequency analysis")
    parser.add_argument("--log_first_rows", action="store_true", help="Log first 5 rows of data from each file")
    parser.add_argument("--rotate", action="store_true", help="Apply rotation based on erotate parameter")
    parser.add_argument("--smoothing_method", default="median",
                        choices=["median", "adaptive"],
                        help="Smoothing method to use: 'median' for median/MAD, 'adaptive' for adaptive median filtering")
    parser.add_argument("--tilt_correction", action="store_true", help="Apply tilt correction so that mean(By) is zero")
    parser.add_argument("--sens_start", type=int, default=0, help="(Unused) Start sample index for sensitivity test")
    parser.add_argument("--sens_end", type=int, default=5000, help="(Unused) End sample index for sensitivity test")
    parser.add_argument("--skip_minutes", type=int, default=0, help="Number of minutes to skip from the beginning of the data")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files instead of displaying them")
    args = parser.parse_args()
    
    processor = ProcessBinary(
        input_dir=args.input_dir,
        param_file=args.param_file,
        average=False,
        perform_freq_analysis=args.perform_freq_analysis,
        plot_data=args.plot_data,
        apply_smoothing=args.apply_smoothing,
        smoothing_window=args.smoothing_window,
        threshold_factor=args.threshold_factor,
        plot_boundaries=args.plot_boundaries,
        plot_smoothed_windows=args.plot_smoothed_windows,
        plot_coherence=args.plot_coherence,
        log_first_rows=args.log_first_rows,
        smoothing_method=args.smoothing_method,
        sens_start=args.sens_start,
        sens_end=args.sens_end,
        skip_minutes=args.skip_minutes
    )
    processor.rotate = args.rotate
    processor.tilt_correction = args.tilt_correction
    processor.save_plots = args.save_plots
    processor.process_all_files()

"""
Example usage:

To process files with median smoothing, tilt correction, skip the first 10 minutes, and save plots:
python ./Stuart_Shelf/Test_Process_06_Smoothing.py --input_dir ./Stuart_Shelf/ST61/merged/ --param_file param.mt --plot_data --apply_smoothing --smoothing_method median --plot_boundaries --plot_smoothed_windows --tilt_correction --skip_minutes 10 --save_plots

To process with adaptive median filtering instead:
python ./Stuart_Shelf/Test_Process_06_Smoothing.py --input_dir ./Stuart_Shelf/ST61/merged/ --param_file param.mt --plot_data --apply_smoothing --smoothing_method adaptive --plot_boundaries --plot_smoothed_windows --tilt_correction --skip_minutes 10 --save_plots
"""