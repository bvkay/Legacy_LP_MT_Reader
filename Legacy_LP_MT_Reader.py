import os
import numpy as np
import pandas as pd
import datetime

LOG_FILE = "process_binary.log"

def write_log(message, level="INFO"):
    """
    Writes log messages to the log file.

    Args:
        message (str): The log message to be written.
        level (str, optional): Log level (INFO, WARNING, ERROR). Defaults to "INFO".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {level} - {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)

class BinaryReader: 
    """
    Reads and processes a single binary file.
    """
    
    def __init__(self, file_path, metadata, average=False, log_first_rows=False):
        """
        Initializes the binary reader.
        
        Args:
            file_path (str): Path to the binary file.
            metadata (dict): Metadata dictionary from the process script.
            average (bool): Whether to average values over intervals.
            log_first_rows (bool): If True, log the first 5 rows of data.
        """
        self.file_path = file_path
        self.metadata = metadata
        self.average = average
        self.log_first_rows = log_first_rows
        self.NCHANNELS = 8
        self.data = None
    
    def read_file(self, outfile):
        """
        Reads a single binary file and writes a brief summary to outfile.
        
        Returns:
            pd.DataFrame: Processed data as a DataFrame.
        """
        try:
            counts = [0] * self.NCHANNELS
            avgcounts = [0] * self.NCHANNELS
            data_rows = []  # Accumulate each row's counts.
            
            with open(self.file_path, "rb") as f:
                # Read header lines and combine into one log entry.
                first_line = f.readline().decode("ascii", errors="ignore").strip()
                sample_rate = int(first_line, 16)
                second_line = f.readline().decode("ascii", errors="ignore").strip()
                filter_point_bytes = f.read(4).decode("ascii", errors="ignore").strip()
                filter_point = int(filter_point_bytes, 16)
                
                header_info = (
                    f"File: {self.file_path}\n"
                    f"Sample Rate: {sample_rate} Hz, Date/Time: {second_line}, "
                    f"Filter Point: {filter_point} (Decimal), {filter_point:X}h (Hex)"
                )

                
                samples_per_second = 10_000_000 / (512 * filter_point)
                samplecount = 0
                first_five_rows = []
                
                # Read data until EOF.
                while True:
                    # Read first 3 channels.
                    for i in range(3):
                        tempchar1 = f.read(1)
                        tempchar2 = f.read(1)
                        tempchar3 = f.read(1)
                        if not tempchar1 or not tempchar2 or not tempchar3:
                            break
                        counts[i] = ((int.from_bytes(tempchar1, byteorder="big", signed=False) << 16) +
                                     (int.from_bytes(tempchar2, byteorder="big", signed=False) << 8) +
                                      int.from_bytes(tempchar3, byteorder="big", signed=False))
                    else:
                        pass
                    if not tempchar1 or not tempchar2 or not tempchar3:
                        break
                    
                    # Read next 2 channels.
                    for i in range(3, 5):
                        tempchar1 = f.read(1)
                        tempchar2 = f.read(1)
                        if not tempchar1 or not tempchar2:
                            break
                        counts[i] = ((int.from_bytes(tempchar1, byteorder="big", signed=False) << 8) +
                                      int.from_bytes(tempchar2, byteorder="big", signed=False))
                    if not tempchar1 or not tempchar2:
                        break
                    
                    # Read channel 5.
                    tempchar1 = f.read(1)
                    if not tempchar1:
                        break
                    counts[5] = int.from_bytes(tempchar1, byteorder="big", signed=False)
                    
                    # Read channels 6 and 7.
                    for i in range(6, 8):
                        tempchar1 = f.read(1)
                        tempchar2 = f.read(1)
                        tempchar3 = f.read(1)
                        if not tempchar1 or not tempchar2 or not tempchar3:
                            break
                        counts[i] = ((int.from_bytes(tempchar1, byteorder="big", signed=False) << 16) +
                                     (int.from_bytes(tempchar2, byteorder="big", signed=False) << 8) +
                                      int.from_bytes(tempchar3, byteorder="big", signed=False))
                    if not tempchar1 or not tempchar2 or not tempchar3:
                        break
                    
                    # Read an extra byte (if present).
                    extra_byte = f.read(1)
                    if not extra_byte:
                        break
                    
                    # Process counts.
                    if self.average:
                        for i in range(self.NCHANNELS):
                            avgcounts[i] += counts[i]
                    else:
                        data_rows.append(counts.copy())
                        row_str = "  ".join(f"{value:08d}" for value in counts)
                        outfile.write(row_str + "\n")
                        if self.log_first_rows and samplecount < 5:
                            first_five_rows.append(row_str)
                    
                    samplecount += 1
                
                # Create DataFrame.
                if not self.average:
                    df = pd.DataFrame(data_rows, columns=[f"chan{i}" for i in range(self.NCHANNELS)])
                else:
                    df = pd.DataFrame([avgcounts], columns=[f"chan{i}" for i in range(self.NCHANNELS)])
                
                if self.log_first_rows and first_five_rows:
                    write_log(header_info)
                    outfile.write(header_info + "\n")
                    write_log("First 5 rows of data:\n" + "\n".join(first_five_rows))
                self.data = df
                return df

        except Exception as e:
            write_log(f"Error reading {self.file_path}: {e}", level="ERROR")