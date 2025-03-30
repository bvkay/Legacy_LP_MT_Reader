# Legacy Adelaide + Flinders Uni Long Period Magnetotelluric Processing

This repository contains a suite of Python scripts for reading the legacy Adelaide and Flinders Uni Long Period MT instruments (commonly referred to as the 'Orange Boxes').

Based upon:
  - Fortran code by G.H., 2003; updated by G.B., 2010
    - File: mt_transform.for 'MT Convertor'
  - C code by W.P., 2001
    - File: 24MAGCON1.C   'Logger Converter for CF1 - PAR24B Boards'


The processing pipeline includes drift correction, unit conversion, optional rotation and tilt correction, and smoothing using either a median/MAD filter or an adaptive median filter. The pipeline also generates plots (power spectra, coherence, and physical channel time series) and saves both raw and processed (scaled) outputs.


## Repository Structure

- **Legacy_LP_MT_Reader.py**  
  Contains the `BinaryReader` class that reads and converts the legacy binary data files into a pandas DataFrame.

- **Legacy_LP_MT_Process.py**  
  Implements the `ProcessBinary` class which:
  - Reads metadata (from a `param.mt` file).
  - Loads binary files (named using the instrument number).
  - Concatenates raw data and builds a continuous time column.
  - Applies drift correction, converts raw counts to physical units, and (optionally) applies rotation and tilt correction.
  - Provides two smoothing options: 
    - **Median/MAD filtering** (the traditional method)
    - **Adaptive median filtering**
  - Generates plots (physical channels, power spectra, coherence) that can either be displayed or saved to files.
  - Saves both the level 0 raw output and the level 1 scaled output with numbers formatted to 5 decimal places.

- **Legacy_LP_MT_Batch.py**  
  Provides a batch processing framework to run the processing pipeline on multiple MT sites in parallel using a `ProcessPoolExecutor`. Each site is assumed to have its own folder (named after the site) within a parent directory. Output files and plots are named based on the site folder name.

> **Note:**  
> When the instrument saves each binary data file, a large spike is often introduced into the time series. This artifact is common in legacy MT data and can affect further analysis if left uncorrected. To address this, our pipeline implements smoothing (via median/MAD or adaptive median filtering) and plots vertical boundary lines to help visually identify these transitions.

- **Stuart_Shelf Data**  
  The repository also contains the Stuart Shelf MT data (binary files and corresponding `param.mt` files) in their respective site folders. These folders are used as input for the processing scripts. This data was collected by the University of Adelaide Geophysics Group in 2009. There's data for four sites (e.g., ST61, ST62, etc.). Each folder contains binary data files (named using the instrument number) and a param.mt file with the metadata. The scripts expect the binary file names to be in the format: [instrument_number]-000.BIN, [instrument_number]-001.BIN, ... and use the instrument_number from the metadata to locate the files.

## Key Features

- **Drift Correction & Time Column:**  
  The pipeline calculates a continuous time column based on the sample interval (e.g., for 10 Hz instruments, sample_interval should be 0.1 seconds) and applies a linear drift correction.

- **Smoothing & Plot Boundaries:**  
  A known artifact in these legacy data is a large spike that appears at the boundaries between binary files—likely due to the file-saving process. To mitigate this:
  - **Smoothing** is applied (using either the median/MAD method or an adaptive median filter) to reduce the impact of these spikes.
  - **Plot boundaries** are added to figures so you can visually identify where the binary files transition.

- **Rotation & Tilt Correction:**  
  Optional rotation corrects for misaligned instruments, and tilt correction adjusts the horizontal magnetic field (By) so its mean is zero.

- **Batch Processing in Parallel:**  
  Process multiple sites concurrently to take advantage of multiple CPU cores. You can control the number of parallel processes via the `--max_workers` option.


## Dependencies

The scripts require Python 3 and the following Python packages:

- numpy
- pandas
- matplotlib
- scipy
- statsmodels


## Usage

### Single-Site Processing
To process a single MT site, run the main processing script. For example, to process data using median smoothing, apply tilt correction, skip the first 10 minutes of data, and save plots, use:

	•	python Test_Process_06_Smoothing.py --input_dir ./Stuart_Shelf/ST61/merged/ --param_file param.mt --plot_data --apply_smoothing --smoothing_method median --plot_boundaries --plot_smoothed_windows --tilt_correction --skip_minutes 10 --save_plots

### Batch Processing (Parallel)
To process multiple MT sites in parallel, use the batch processing script. For example, to process sites ST61, ST62, ST63, ST64, and ST67 with rotation, tilt correction, median smoothing, skipping the first 10 minutes, and saving all plots, run:

	•	python Test_Batch_Process_Parallel.py --parent_dir ./Stuart_Shelf/ --sites ST61 ST62 ST63 ST64 ST67 --param_file param.mt --rotate --tilt_correction --apply_smoothing --plot_data --plot_boundaries --plot_smoothed_windows --smoothing_method median --skip_minutes 10 --save_plots --max_workers 4

### Command-Line Options

- input_dir: Directory containing the binary files and param.mt for a single site.
- parent_dir: Parent directory containing all MT site folders (used in batch processing).
- sites: List of site folder names to process.
- param_file: Name of the parameter file (e.g., param.mt).
- rotate: Apply rotation correction based on the instrument’s erotate parameter.
- tilt_correction: Apply tilt correction so that the mean of the horizontal magnetic field (By) is zero.
- apply_smoothing: Apply smoothing to the data.
- smoothing_method: Choose the smoothing method: median for median/MAD filtering or adaptive for adaptive median filtering.
- smoothing_window: Window size for smoothing.
- threshold_factor: Threshold multiplier for outlier detection.
- plot_data: Plot the physical channel data.
- plot_boundaries: Draw vertical lines at file boundaries on the plots.
- plot_smoothed_windows: Shade regions where smoothing was applied.
- plot_coherence: Plot power spectra and coherence.
- perform_freq_analysis: Perform FFT-based frequency analysis.
- log_first_rows: Log the first 5 rows from each binary file (for debugging).
- skip_minutes: Number of minutes to skip from the beginning of the data.
- save_plots: Save plots to files instead of displaying them.
- max_workers: Maximum number of parallel processes to use during batch processing.

### Data Details

The Stuart Shelf data is organized in site-specific folders (e.g., ST61, ST62, etc.) within a parent directory. Each folder contains:
- Binary files named with the instrument number followed by a sequential index (e.g., HFM4-000.BIN, HFM4-001.BIN, etc.).
- A param.mt file that contains metadata, including the instrument number, electrode separations, start/finish times, sample interval, and time drift.
