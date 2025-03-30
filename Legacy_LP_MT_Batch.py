#!/usr/bin/env python3
"""
Batch Process MT Data in Parallel

This script processes multiple magnetotelluric (MT) sites in batch mode.
Each site is assumed to be in its own folder (named after the site) contained within a parent directory.
For each site, a ProcessBinary instance (from Test_Process_06_Smoothing.py) is created and run in parallel
using a ProcessPoolExecutor. Output files and figures are named using the site folder name.

Usage example:
    python ./Stuart_Shelf/Legacy_LP_MT_Batch.py --parent_dir ./Stuart_Shelf/ --sites ST61 ST62 ST63 ST64 --param_file param.mt --rotate --tilt_correction --apply_smoothing --plot_data --plot_boundaries --plot_smoothed_windows --smoothing_method median --skip_minutes 10 --save_plots --max_workers 4
"""

import os
import argparse
import concurrent.futures
from Legacy_LP_MT_Process import ProcessBinary, write_log

def process_single_site(site, parent_dir, param_file, rotate, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots):
    """
    Processes a single site folder.
    
    Args:
        site (str): Site folder name.
        parent_dir (str): Parent directory containing the site folders.
        param_file (str): Parameter file name.
        rotate (bool): Whether to apply rotation.
        tilt_correction (bool): Whether to apply tilt correction.
        apply_smoothing (bool): Whether to apply smoothing.
        smoothing_method (str): Smoothing method to use ("median" or "adaptive").
        smoothing_window (int): Window size for smoothing.
        threshold_factor (float): Threshold multiplier for outlier detection.
        plot_data (bool): Whether to plot physical channel data.
        plot_boundaries (bool): Whether to show file boundaries on plots.
        plot_smoothed_windows (bool): Whether to shade smoothed windows.
        plot_coherence (bool): Whether to plot power spectra and coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log the first 5 rows from each file.
        sens_start (int): Start sample index for sensitivity test (unused now).
        sens_end (int): End sample index for sensitivity test (unused now).
        skip_minutes (int): Number of minutes to skip from the beginning.
        save_plots (bool): If True, plots are saved to disk.
    
    Returns:
        str: Site name upon completion.
    """
    site_dir = os.path.join(parent_dir, site)
    if not os.path.isdir(site_dir):
        write_log(f"Site folder not found: {site_dir}", level="ERROR")
        return None

    write_log(f"Starting processing for site: {site}")
    processor = ProcessBinary(
        input_dir=site_dir,
        param_file=param_file,
        average=False,
        perform_freq_analysis=perform_freq_analysis,
        plot_data=plot_data,
        apply_smoothing=apply_smoothing,
        smoothing_window=smoothing_window,
        threshold_factor=threshold_factor,
        plot_boundaries=plot_boundaries,
        plot_smoothed_windows=plot_smoothed_windows,
        plot_coherence=plot_coherence,
        log_first_rows=log_first_rows,
        smoothing_method=smoothing_method,
        sens_start=sens_start,
        sens_end=sens_end,
        skip_minutes=skip_minutes
    )
    processor.rotate = rotate
    processor.tilt_correction = tilt_correction
    processor.save_plots = save_plots
    processor.process_all_files()
    write_log(f"Completed processing for site: {site}")
    return site


def batch_process_sites(sites, parent_dir, param_file, rotate, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        max_workers):
    """
    Processes multiple MT sites in parallel.
    
    Args:
        sites (list): List of site folder names.
        parent_dir (str): Parent directory containing the site folders.
        max_workers (int): Maximum number of parallel processes to use.
        (Other arguments are passed to process_single_site)
    
    Returns:
        None
    """
    write_log("\n\nStarting batch processing of sites: \n" + "\n".join(sites))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_site, site, parent_dir, param_file, rotate, tilt_correction,
                apply_smoothing, smoothing_method, smoothing_window, threshold_factor,
                plot_data, plot_boundaries, plot_smoothed_windows, plot_coherence,
                perform_freq_analysis, log_first_rows, sens_start, sens_end, skip_minutes, save_plots
            )
            for site in sites
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                write_log(f"Batch processing: Completed site {result}")
    write_log("\nBatch processing completed.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process multiple MT sites contained in individual folders in parallel."
    )
    parser.add_argument("--parent_dir", required=True,
                        help="Parent directory containing MT site folders.")
    parser.add_argument("--sites", nargs="+", required=True,
                        help="List of site folder names to process.")
    parser.add_argument("--param_file", default="param.mt",
                        help="Parameter file name located in each site folder.")
    parser.add_argument("--rotate", action="store_true",
                        help="Apply rotation correction based on the 'erotate' parameter.")
    parser.add_argument("--tilt_correction", action="store_true",
                        help="Apply tilt correction so that mean(By) is zero.")
    parser.add_argument("--apply_smoothing", action="store_true",
                        help="Apply smoothing to the data.")
    # Only "median" and "adaptive" are allowed smoothing methods now.
    parser.add_argument("--smoothing_method", default="median",
                        choices=["median", "adaptive"],
                        help="Smoothing method to use: 'median' for median/MAD or 'adaptive' for adaptive median filtering.")
    parser.add_argument("--smoothing_window", type=int, default=2500,
                        help="Window size for smoothing.")
    parser.add_argument("--threshold_factor", type=float, default=10.0,
                        help="Threshold multiplier for outlier detection (for median method).")
    parser.add_argument("--plot_data", action="store_true",
                        help="Plot physical channel data.")
    parser.add_argument("--plot_boundaries", action="store_true",
                        help="Plot file boundaries on the plots.")
    parser.add_argument("--plot_smoothed_windows", action="store_true",
                        help="Shade smoothed windows in the plots.")
    parser.add_argument("--plot_coherence", action="store_true",
                        help="Plot power spectra and coherence.")
    parser.add_argument("--perform_freq_analysis", action="store_true",
                        help="Perform frequency analysis on the data.")
    parser.add_argument("--log_first_rows", action="store_true",
                        help="Log the first 5 rows of data from each binary file.")
    parser.add_argument("--sens_start", type=int, default=0,
                        help="(Unused) Start sample index for sensitivity test.")
    parser.add_argument("--sens_end", type=int, default=5000,
                        help="(Unused) End sample index for sensitivity test.")
    parser.add_argument("--skip_minutes", type=int, default=0,
                        help="Number of minutes to skip from the beginning of the data.")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to files instead of displaying them.")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of parallel processes to use.")

    args = parser.parse_args()

    batch_process_sites(
        sites=args.sites,
        parent_dir=args.parent_dir,
        param_file=args.param_file,
        rotate=args.rotate,
        tilt_correction=args.tilt_correction,
        apply_smoothing=args.apply_smoothing,
        smoothing_method=args.smoothing_method,
        smoothing_window=args.smoothing_window,
        threshold_factor=args.threshold_factor,
        plot_data=args.plot_data,
        plot_boundaries=args.plot_boundaries,
        plot_smoothed_windows=args.plot_smoothed_windows,
        plot_coherence=args.plot_coherence,
        perform_freq_analysis=args.perform_freq_analysis,
        log_first_rows=args.log_first_rows,
        sens_start=args.sens_start,
        sens_end=args.sens_end,
        skip_minutes=args.skip_minutes,
        save_plots=args.save_plots,
        max_workers=args.max_workers
    )
"""
To run: 

python ./Stuart_Shelf/Test_Batch_Process.py --input_dir ./Stuart_Shelf/ --sites ST61 ST62 ST63 ST64 ST67 --param_file param.mt --rotate --tilt_correction --apply_smoothing --plot_boundaries --plot_smoothed_windows --smoothing_method median --save_plots --plot_data --plot_boundaries

python ./Stuart_Shelf/Test_Batch_Process_Parallel.py --input_dir ./Stuart_Shelf/ --sites ST61 ST62 ST63 ST64 ST67 --param_file param.mt --max_workers 4 --rotate --tilt_correction --apply_smoothing --skip_minutes 20 --plot_data --plot_boundaries --plot_smoothed_windows --smoothing_method median --save_plots

"""