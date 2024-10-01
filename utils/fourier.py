import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os


# Define a function to apply Fourier transform on a time window
def apply_fft(window, sampling_rate=50):  
    n = len(window)  # Number of samples in the window
    yf = fft(window)  # Apply FFT
    xf = fftfreq(n, 1 / sampling_rate)  # Frequency domain

    # Return the positive half of the spectrum (frequencies and corresponding amplitudes)
    # because FFT results are symmetrical for real-valued signals
    freqs = xf[:n // 2]   # Positive half of frequencies
    amplitudes = np.abs(yf[:n // 2])  # Magnitudes of the FFT coefficients

    return freqs, amplitudes


def apply_fft_df(df, file_name, window_duration=1, sampling_rate=50, column='accel.mag', plot=False):
    # Split the data into windows
    nrows = len(df)
    window_size = window_duration * sampling_rate
    n_windows = nrows // window_size
    df['Window'] = np.repeat(np.nan, nrows)
    window_col_index = df.columns.get_loc("Window")

    # Initialize an empty list to store the DataFrames for each window
    dfs = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window = df.iloc[start_idx: end_idx][column]
        df.iloc[start_idx: end_idx, window_col_index] = np.repeat(i, window_size)
        # Apply FFT on the current window
        freqs, amplitudes = apply_fft(window.dropna(), sampling_rate=50)  # Adjust sampling rate

        # Store frequencies and amplitudes into a DataFrame
        window_df = pd.DataFrame({
            'Frequency_Hz': freqs,
            'Amplitude': amplitudes,
            'Window': i,
            "file_name": file_name
        })

        # Append the DataFrame to the list
        dfs.append(window_df)

        if plot:
            # Plot the frequency spectrum for the current window
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))

            axs[0].plot(freqs, amplitudes)
            axs[0].set_xlabel('Frequency (Hz)')
            axs[0].set_ylabel('Amplitude')
            axs[0].grid(True)

            axs[1].plot(window.index, window, label='Acceleration magnitude', color='r')
            axs[1].set_ylabel('')
            axs[1].grid(True)

            plt.suptitle(f"Frequency Spectrum for Window {i+1}")
            plt.show()

    # Concatenate all the individual DataFrames into one final DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    return final_df


def apply_fft_df_batch(import_folder_path, keyword='standstill', window_duration=1, sampling_rate=50, plot=False):
    """
    Import all csv files in the folder and split them into one minute chunks.
    """
    csv_files = [f for f in os.listdir(import_folder_path) if 'csv' in f and keyword in f]
    key_coloumns = ['timestamp', 'accel.x', 'accel.y', 'accel.z']
    # Initialize an empty list to store the result for each DataFrames
    dfs = []

    for f in csv_files:
        df = pd.read_csv(import_folder_path + f)
        # Drop the first and last 500 rows (10 seconds) to remove noises caused by button pressings
        df = df.iloc[500:-500].reset_index()

        if all(col in df.columns for col in key_coloumns):
            print(f'{f} contains needed columns. Processing...')
            # Drop the first and last 500 rows (10 seconds) to remove noises caused by button pressings
            df = df.iloc[500:-500].reset_index()
            # Create a new column for acceleration magnitude
            df['accel.mag'] = np.sqrt(df['accel.x']**2 + df['accel.y']**2 + df['accel.z']**2)
            # Calculate the FFT for each 1 minute chunk
            df_result = apply_fft_df(df, file_name=f, window_duration=window_duration, sampling_rate=sampling_rate, column='accel.mag', plot=plot)
            dfs.append(df_result)
        else:
            print(f'{f} does not contain needed columns. Skipping...')

    final_df = pd.concat(dfs, ignore_index=True)

    return final_df
