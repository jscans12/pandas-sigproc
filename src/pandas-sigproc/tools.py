# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy.signal
import warnings
from scipy.io import wavfile


def build_freq_array(fn_start: float = 10., fn_end: float = 1000.,
                     oct_step_size: float = (1. / 12.)):
    """
    Get an array of natural frequencies
    
    @author: dsholes
    Date: November 8, 2018
    Version: 0.1
    https://github.com/dsholes/python-srs
    
    Parameters
    ----------
    fn_start : float, optional
        Start frequency. The default is 10.
    fn_end : float, optional
        End frequency. The default is 1000.
    oct_step_size : float, optional
        Octave step size. The default is 1/12.
    
    Returns
    -------
    float array
        The frequency array
    
    """

    fn_array = [fn_start]
    for i in range(int(fn_end - fn_start)):
        new_fn = (fn_start * 2. ** oct_step_size)
        fn_array.append(new_fn)
        fn_start = new_fn
        if fn_start > fn_end:
            break
    fn_array = np.array(fn_array)

    return fn_array


def psd(values, sample_rate,
        window_length: float = None, overlap: float = 0.5, window_type: str = 'hann',
        summarize: str = 'mean', fatigue_exponent: float = 4.0,
        detrend='constant'):
    """
    Compute a power spectral density from the timedomain. Uses periodogram by default,
    but user can specify a window and use Welch's method by default. There are also other
    methods offered to perform statistics on the various sub-PSDs.
    https://en.wikipedia.org/wiki/Spectral_density

    Parameters
    ----------
    values
        Array of samples
    sample_rate
        The sample rate of the signal
    window_length : float, optional
        Window length in units time for each sub-PSD. The default is None.
    overlap : float, optional
        Overlap ratio for windows. The default is 0.5.
    window_type : str, optional
        Window function. See scipy.signal.get_window for options. The default is 'hann'.
    summarize : str, optional
        Method for averaging periodograms if using Welch's method. Can be
        'mean', 'median', 'max', 'min', 'max rms', 'damage', or 'all'. 'all' returns all sub-PSDs.
        'max rms' returns the mean scaled to the max observed RMS over sub-PSDs. 'damage' uses the
        exponential fatigue damage relationship to scale the PSD for non-stationary processes.
        The default is 'mean'.
    fatigue_exponent : float
        Fatigue exponent when using 'damage' summary method.
    detrend : optional
        Detrend method- constant, linear or False for no detrend

    Returns
    -------
    freq_out
        Frequency vector
    psd_out
        Power spectral density

    """

    # Number of samples for window
    n_window = int(sample_rate * window_length)

    # Number of samples to overlap
    n_overlap = round(n_window * overlap)

    # Number of samples overall
    n_total = values.size

    # Step size
    n_step = n_window - n_overlap

    # Create window array
    shape = values.shape[:-1] + ((values.shape[-1] - n_overlap) // n_step, n_window)
    strides = values.strides[:-1] + (n_step * values.strides[-1], values.strides[-1])
    windows = np.lib.stride_tricks.as_strided(values, shape=shape,
                                              strides=strides)

    # Compute sub-PSDs
    freq_out, psd_all = scipy.signal.periodogram(windows, fs=sample_rate, window=window_type, detrend=detrend, axis=1)

    # Statistic over sub-PSDs
    if summarize == 'mean':
        psd_out = psd_all.mean(axis=0)
    elif summarize == 'median':
        psd_out = np.median(psd_all, axis=0)
    elif summarize == 'max':
        psd_out = psd_all.max(axis=0)
    elif summarize == 'min':
        psd_out = psd_all.min(axis=0)
    elif summarize == 'all':
        psd_out = psd_all
    elif summarize == 'max rms':
        psd_out = psd_all.mean(axis=0)
        mean_rms = rms_psd_linear(psd_out, freq_out)
        rms_list = np.apply_along_axis(rms_psd_linear, 1, psd_all, freq_out)
        max_rms = rms_list.max()
        psd_out = psd_out * (max_rms / mean_rms) ** 2
    elif summarize == 'damage':
        psd_out = psd_all.mean(axis=0)
        mean_rms = rms_psd_linear(psd_out, freq_out)
        rms_list = np.apply_along_axis(rms_psd_linear, 1, psd_all, freq_out)
        acceleration_factor = np.mean((rms_list / mean_rms) ** fatigue_exponent) ** (1/fatigue_exponent)
        psd_out = psd_out * acceleration_factor ** 2
    else:
        raise Exception('%s is not a valid summary type' % summarize)

    return freq_out, psd_out


def rms(values):
    """
    Get the root-mean-square (RMS) of a vector
    @author: john.scanlon

    Parameters
    ----------
    values
        Array of values over which to compute the RMS

    Returns
    -------
    float
        RMS

    """

    return np.sqrt(np.mean(values ** 2))


def rms_psd_linear(power, freq):
    """
    Get the RMS of the PSD in linear space. This is suitable for PSDs which
    are densely populated, such as PSDs calculated from a time history.
    @author: john.scanlon

    Parameters
    ----------
    power
        PSD spectrum
    freq
        Frequency vector

    Returns
    -------
    float
        RMS

    """

    return np.sqrt(np.trapz(power, x=freq))


def rms_psd_loglog(power, freq):
    """
    Get the RMS of the PSD in log-log space, recommended for sparsely
    populated PSDs which have linear regions on a log-log plot. Typically
    specs fall into this category. Densely populated PSDs will likely run
    out of machine precision and need to revert to linear space for the
    integral.
    @author: john.scanlon

    Parameters
    ----------
    power
        PSD spectrum
    freq
        Frequency vector

    Returns
    -------
    float
        The RMS value

    """

    # I know this code will cause warnings but don't want to see them.
    # There is a good fallback method here to deal with them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Need to perform some fancy logspace integration from:
        # http://www.vibrationdata.com/tutorials_alt/psdinteg.pdf
        ms = 0
        flag = True
        for i in range(0, freq.size - 1):

            # Compute n from the source above
            n = np.log10(power[i + 1] / power[i]) / np.log10(freq[i + 1] / freq[i])

            # Attempt to find the area over log space
            if abs(n + 1.) < 1e-10:
                i_area = (power[i] * freq[i]) * np.log(freq[i + 1] / freq[i])
            else:
                i_area = ((power[i] / (freq[i] ** n)) * (1 / (n + 1)) *
                          (freq[i + 1] ** (n + 1) - freq[i] ** (n + 1)))

            # If too small for machine precision, find in linear space
            if np.isnan(i_area) or i_area == 0 or np.isinf(i_area):
                if flag:
                    print('Reverted to linear algorithm for some freq \
                                   steps due to insufficient machine precision')
                    flag = False
                i_area = ((power[i] + power[i + 1]) / 2) * (freq[i + 1] - freq[i])

            # Add to the mean square
            ms += i_area

    return np.sqrt(ms)


def spl(values, p_ref):
    """
    Get the sound pressure level of a vector
    @author: john.scanlon

    Parameters
    ----------
    values
        Array of values over which to compute the RMS
    p_ref
        Reference pressure

    Returns
    -------
    float
        SPL

    """

    values_rms = rms(values)
    return 20*np.log10(values_rms/p_ref)


def a_weighting(fs):
    """
    Design of an A-weighting filter.
    b, a = a_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).

    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    Translated from a MATLAB script (which also includes C-weighting, octave
    and one-third-octave digital filters).

    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
        couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.

    """

    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    a1000 = 1.9997
    pi = np.pi

    # Compute filter coefficients
    nums = [(2*pi * f4)**2 * (10**(a1000/20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4*pi * f4, (2*pi * f4)**2],
                      [1, 4*pi * f1, (2*pi * f1)**2])
    dens = np.polymul(np.polymul(dens, [1, 2*pi * f3]),
                      [1, 2*pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return scipy.signal.bilinear(nums, dens, fs)


def write_wav(filename, sample_rate, sound_left, sound_right=None):
    """
    Create a stereo or mono wav file
    """

    # Copy left channel for mono files
    if sound_right is None:
        sound_right = sound_left

    # A 2D array where the left and right tones are contained in their respective columns
    tone_y_stereo = np.vstack((sound_left, sound_right))
    tone_y_stereo = tone_y_stereo.transpose()

    # Produce an audio file that contains stereo sound
    wavfile.write(filename, sample_rate, tone_y_stereo)

    
def srs(time: np.array, accel: np.array, fn_array: np.array = None,
        quality_factor: float = 50., remove_bias: bool = False):
    """
    Compute a shock response spectrum
    https://en.wikipedia.org/wiki/Shock_response_spectrum
    
    @author: dsholes
    Date: November 8, 2018
    Version: 0.1
    https://github.com/dsholes/python-srs
    
    Review Smallwood method in his paper:
        - 'AN IMPROVED RECURSIVE FORMULA FOR CALCULATING SHOCK RESPONSE SPECTRA'
        - https://www.vibrationdata.com/ramp_invariant/DS_SRS1.pdf
        
    Parameters
    ----------
    time : numpy array
        Timestamp vector
    accel : numpy array
        Acceleration vector
    fn_array : float array, optional
        Frequency vector over which the SRS will be calculated. The default is
        the defaults for build_freq_array.
    quality_factor : float, optional
        Quality factor. The default is 50.
    remove_bias : bool
        Remove static acceleration bias from the data
    
    Returns
    -------
    numpy array
        The max absolute acceleration of an SDOF at each natural frequency in
        the supplied fn_array
        
    """

    # Default for freq array
    if fn_array is None:
        fn_array = build_freq_array()

    if remove_bias:
        accel = accel - accel.mean()
        print('Input data has been modified to remove sensor bias (offset)...')

    # Should I give user access to the following coefficients??
    damp = 1. / (2. * quality_factor)
    t = np.diff(time).mean()  # sample interval
    omega_n = 2. * np.pi * fn_array
    omega_d = omega_n * np.sqrt(1 - damp ** 2.)
    e = np.exp(-damp * omega_n * t)
    k = t * omega_d
    c = e * np.cos(k)
    s = e * np.sin(k)
    s_prime = s / k
    b0 = 1. - s_prime
    b1 = 2. * (s_prime - c)
    b2 = e ** 2. - s_prime
    a0 = np.ones_like(fn_array)  # Necessary because of how scipy.signal.lfilter() is structured
    a1 = -2. * c
    a2 = e ** 2.
    b = np.array([b0, b1, b2]).T
    a = np.array([a0, a1, a2]).T

    # Calculate SRS using Smallwood ramp invariant method
    pos_accel = np.zeros_like(fn_array)
    neg_accel = np.zeros_like(fn_array)
    for i, f_n in enumerate(fn_array):
        output_accel_g = scipy.signal.lfilter(b[i], a[i], accel)
        pos_accel[i] = output_accel_g.max()
        neg_accel[i] = np.abs(output_accel_g.min())

    return pos_accel, neg_accel


def build_shock_pulse(pulse_width: float, time_dt: float,
                      mag: float = 1.,
                      pre_padding: float = 1.,
                      post_padding: float = 10.,
                      waveform: str = 'half sine'):
    """
    Build a shock pulse
    @author: john.scanlon
    
    Parameters
    ----------
    pulse_width : float
        Pulse width in s
    time_dt :float
        Time step in s
    mag : float, optional
        Shock magnitude. The default is 1.
    pre_padding : float, optional
        Padding prior to the shock pulse in multiples of pulse width. The default
        is 1.
    post_padding : float, optional
        Padding after the shock pulse in multiples of pulse width. The default
        is 10.
    waveform : str, optional
        The waveform of the shock pulse. Options are typical electrodynamic shaker waveforms-
        half sine, haversine, square, triangular, initial sawtooth, terminal sawtooth
        The default is half sine.
        
    Returns
    -------
    time : float array
        Time vector
    accel : float array
        Shock acceleration vector
    
    """

    # End time for the pulse history
    time_end = pulse_width * (pre_padding + post_padding + 1)

    # Create a time vector
    time = np.linspace(0, time_end, round((time_end / time_dt) + 1))

    # Generate the shock pulse
    if waveform == 'half sine' or waveform == 'sine' or waveform == 'sin':
        accel = mag * np.sin((np.pi * (time - pulse_width * pre_padding)) / pulse_width)
    elif waveform == 'haversine':
        accel = mag * np.sin((np.pi * (time-pulse_width*pre_padding)) / pulse_width) ** 2
    elif waveform == 'square' or waveform == 'rectangular' or waveform == 'rectangle' or waveform == 'rect':
        accel = mag * np.ones_like(time)
    elif waveform == 'triangular' or waveform == 'triangle':
        # This is probably needlessly overcomplicated...
        accel = mag * np.zeros_like(time)
        m1 = (2*mag) / pulse_width
        b1 = (-2*mag*pulse_width*pre_padding) / pulse_width
        m2 = -(2*mag) / pulse_width
        b2 = (2*mag*(pulse_width*(pre_padding + 1))) / pulse_width
        accel[time < pulse_width * (pre_padding + 0.5)] = m1 * time[time < pulse_width * (pre_padding + 0.5)] + b1
        accel[time >= pulse_width * (pre_padding + 0.5)] = m2 * time[time >= pulse_width * (pre_padding + 0.5)] + b2
    elif waveform == 'initial sawtooth' or waveform == 'init sawtooth':
        m = -mag / pulse_width
        b = (mag*pulse_width*(1+pre_padding)) / pulse_width
        accel = m * time + b
    elif waveform == 'terminal sawtooth' or waveform == 'term sawtooth':
        m = mag / pulse_width
        b = (-mag * pulse_width * pre_padding) / pulse_width
        accel = m * time + b
    else:
        raise Exception('%s is not a defined waveform' % waveform)

    # Zero the padding ranges
    accel[time < pulse_width * pre_padding] = 0
    accel[time > pulse_width * (pre_padding + 1)] = 0

    return time, accel
  
