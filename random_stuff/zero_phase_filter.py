import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import filtfilt, firwin

# --- Part 1: Core Filtering Logic (From Scratch using FFTs) ---

def fir_filter_with_fft(signal, kernel):
    """
    Applies a linear-phase FIR filter to a signal using the FFT method
    (fast convolution). This introduces a delay.

    Args:
        signal (np.ndarray): The input signal to be filtered.
        kernel (np.ndarray): The impulse response of the FIR filter.

    Returns:
        np.ndarray: The filtered signal, with group delay.
    """
    # For linear convolution using FFTs, the required length is N + M - 1,
    # where N is the signal length and M is the kernel length.
    signal_len = len(signal)
    kernel_len = len(kernel)
    final_len = signal_len + kernel_len - 1

    # 1. Compute the FFT of the signal, padded to the final length
    signal_fft = fft(signal, n=final_len)

    # 2. Compute the FFT of the kernel, padded to the final length
    kernel_fft = fft(kernel, n=final_len)

    # 3. Perform element-wise multiplication in the frequency domain
    filtered_fft = signal_fft * kernel_fft

    # 4. Compute the inverse FFT to get the filtered signal in the time domain
    filtered_signal = ifft(filtered_fft)

    # Trim the result of the convolution back to the original signal length
    return np.real(filtered_signal[:signal_len])


def zero_phase_fir_filter(signal, kernel):
    """
    Applies a zero-phase FIR filter by manually compensating for the 
    kernel's group delay. This is suitable for offline processing.

    Args:
        signal (np.ndarray): The input signal.
        kernel (np.ndarray): The symmetric FIR filter kernel.

    Returns:
        np.ndarray: The zero-phase filtered signal.
    """
    # 1. Filter the signal using the standard FFT convolution method
    filtered_signal = fir_filter_with_fft(signal, kernel)

    # 2. Calculate the group delay of the symmetric FIR kernel
    kernel_len = len(kernel)
    delay = (kernel_len - 1) // 2

    # 3. Compensate for the delay by shifting the output signal to the left
    # The `np.roll` function shifts the elements. The last 'delay' samples
    # are artifacts from the beginning of the signal, so we clear them.
    zero_phase_filtered = np.roll(filtered_signal, -delay)
    zero_phase_filtered[-delay:] = 0

    return zero_phase_filtered


# --- Part 2: FIR Filter Kernel Generation (From Scratch) ---

def create_lowpass_kernel(cutoff_hz, fs, num_taps):
    """
    Creates a windowed-sinc low-pass FIR filter kernel.

    Args:
        cutoff_hz (float): The cutoff frequency in Hz.
        fs (int): The sampling frequency in Hz.
        num_taps (int): The number of taps. Should be an odd number.

    Returns:
        np.ndarray: The low-pass filter kernel.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd number for a symmetric kernel.")
    
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    t = np.arange(num_taps)
    center = (num_taps - 1) / 2
    x = t - center
    sinc_kernel = np.sinc(x * normalized_cutoff)
    window = np.blackman(num_taps)
    windowed_sinc = sinc_kernel * window
    return windowed_sinc / np.sum(windowed_sinc)


def create_highpass_kernel(cutoff_hz, fs, num_taps):
    """
    Creates a high-pass FIR filter kernel using spectral inversion.

    Args:
        cutoff_hz (float): The cutoff frequency in Hz.
        fs (int): The sampling frequency in Hz.
        num_taps (int): The number of taps. Should be odd.

    Returns:
        np.ndarray: The high-pass filter kernel.
    """
    lp_kernel = create_lowpass_kernel(cutoff_hz, fs, num_taps)
    center = (num_taps - 1) / 2
    impulse = np.zeros(num_taps)
    impulse[int(center)] = 1.0
    hp_kernel = impulse - lp_kernel
    return hp_kernel


# --- Part 3: Demonstration and Validation ---

if __name__ == '__main__':
    # --- Setup ---
    fs = 1000
    duration = 2
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    low_freq = 20
    high_freq = 200
    signal = (np.sin(2 * np.pi * low_freq * t) +
              0.5 * np.sin(2 * np.pi * high_freq * t))
    num_taps = 101
    cutoff_hz = 80

    # --- 1. Low-Pass Filtering Demo (Zero-Phase) ---
    print("--- Testing Zero-Phase Low-Pass Filter ---")

    lp_kernel_scratch = create_lowpass_kernel(cutoff_hz, fs, num_taps)
    # Filter using our new zero-phase function
    lp_filtered_scratch = zero_phase_fir_filter(signal, lp_kernel_scratch)

    # For validation: use SciPy's zero-phase filtfilt function
    lp_kernel_scipy = firwin(num_taps, cutoff_hz, fs=fs, pass_zero='lowpass')
    lp_filtered_scipy = filtfilt(lp_kernel_scipy, 1.0, signal)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Zero-Phase Low-Pass Filtering (Cutoff = {cutoff_hz} Hz)', fontsize=16)

    axs[0].plot(t, signal, label='Original Signal')
    axs[0].set_title(f'Original Signal ({low_freq} Hz + {high_freq} Hz)')
    axs[0].legend()

    axs[1].plot(t, lp_filtered_scratch, label='Filtered (Scratch Zero-Phase)', color='r')
    axs[1].set_title('Filtered with Scratch Implementation (Phase Corrected)')
    axs[1].legend()

    axs[2].plot(t, lp_filtered_scipy, label='Filtered (SciPy filtfilt)', color='g')
    axs[2].set_title('Filtered with SciPy for Validation (Zero-Phase)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- 2. High-Pass Filtering Demo (Zero-Phase) ---
    print("\n--- Testing Zero-Phase High-Pass Filter ---")

    hp_kernel_scratch = create_highpass_kernel(cutoff_hz, fs, num_taps)
    hp_filtered_scratch = zero_phase_fir_filter(signal, hp_kernel_scratch)

    # For validation: use SciPy's zero-phase filtfilt function
    hp_kernel_scipy = firwin(num_taps, cutoff_hz, fs=fs, pass_zero='highpass')
    hp_filtered_scipy = filtfilt(hp_kernel_scipy, 1.0, signal)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Zero-Phase High-Pass Filtering (Cutoff = {cutoff_hz} Hz)', fontsize=16)

    axs[0].plot(t, signal, label='Original Signal')
    axs[0].set_title(f'Original Signal ({low_freq} Hz + {high_freq} Hz)')
    axs[0].legend()

    axs[1].plot(t, hp_filtered_scratch, label='Filtered (Scratch Zero-Phase)', color='r')
    axs[1].set_title('Filtered with Scratch Implementation (Phase Corrected)')
    axs[1].legend()

    axs[2].plot(t, hp_filtered_scipy, label='Filtered (SciPy filtfilt)', color='g')
    axs[2].set_title('Filtered with SciPy for Validation (Zero-Phase)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # --- 3. Plotting Filter Frequency Responses ---
    N_FFT = 2048
    freq_axis = fftfreq(N_FFT, 1/fs)
    lp_kernel_fft = fft(lp_kernel_scratch, N_FFT)
    hp_kernel_fft = fft(hp_kernel_scratch, N_FFT)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis[:N_FFT//2], 20 * np.log10(np.abs(lp_kernel_fft[:N_FFT//2])), label='Low-Pass Kernel')
    plt.plot(freq_axis[:N_FFT//2], 20 * np.log10(np.abs(hp_kernel_fft[:N_FFT//2])), label='High-Pass Kernel', linestyle='--')
    plt.axvline(cutoff_hz, color='k', linestyle=':', label=f'Cutoff Freq: {cutoff_hz} Hz')
    plt.title('Frequency Response of Generated Filter Kernels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-60, 5)
    plt.legend()
    plt.grid(True)
    plt.show()