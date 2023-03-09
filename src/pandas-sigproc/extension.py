# -*- coding: utf-8 -*-
"""
Extensions of Pandas series
@author: john.scanlon
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.integrate
import scipy.interpolate
import pandas-sigproc.tools as sp_tools
import sounddevice as sd
import rainflow as rf


@pd.api.extensions.register_series_accessor("timedomain")
class TimeDomain:
    """
    Class for working with time series data
    """

    _obj = None
    "Object property"

    unit = "none"
    "Engineering units"

    def __init__(self, pd_series: pd.Series):
        """
        Class constructor

        Parameters
        ----------
        pd_series : pandas Series
            Pandas series object

        """

        # Construct in the superclass
        pd_series.index.name = "time"
        self._obj = pd_series

    @property
    def _name(self):
        """Name of the series"""
        return str(self._obj.name)

    @property
    def samplerate(self):
        """Sample rate of the time series (computed from time steps)"""

        # Get dt for each time step
        x = self._get_x()
        diff_x = np.diff(x)

        # Different stats on samplerate
        max_sr = 1 / np.amax(diff_x)
        mean_sr = 1 / np.mean(diff_x)
        min_sr = 1 / np.amin(diff_x)

        # Test if within a reasonable tolerance, if not either this method is
        # weak or the signal was logged with a variable sample rate
        sr_error = (max_sr - min_sr) / min_sr
        variability_tol = 0.0001
        sr = mean_sr
        if sr_error > (min_sr * variability_tol) or np.isnan(sr_error):
            sr = None

        # Convert to integer if possible, assume the sample rate must be integer if it's within integer_tol of an
        # integer and also greater than 1 Hz
        integer_tol = 0.01
        if abs(sr - round(sr)) < integer_tol and sr >= 1:
            sr = int(sr)

        return sr

    @property
    def _start_time(self):
        """Get the start time"""

        if self._obj.index.inferred_type == "datetime64":
            start_time = self._obj.index.values[0]
        elif self._obj.index.inferred_type == "floating":
            start_time = None
        else:
            raise NotImplementedError

        return start_time

    def _get_x(self):
        """Get the x values"""

        if self._obj.index.inferred_type == "datetime64":
            t_seconds = (self._obj.index.values - self._obj.index.values[0]) / np.timedelta64(1, 's')
        elif self._obj.index.inferred_type == "floating":
            t_seconds = self._obj.index.to_numpy(copy=True)
        else:
            raise NotImplementedError

        return t_seconds

    def _get_y(self):
        """Get the y values"""
        return self._obj.to_numpy(copy=True)

    def plot(self, ax: plt.axis = None, **kwargs):
        """
        Plot the timedomain

        Parameters
        ----------
        ax : pyplot axis, optional
            Pass an existing axis on which to draw the plot. The default is
            None.
        **kwargs
            Optional arguments for pyplot

        Returns
        -------
        ax : pyplot axis
            The axis used for this plot

        """

        # Create a new figure and axis if one isn't supplied
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Basic plotting
        ax.plot(x, y, label=self._name, **kwargs)

        # Formatting
        ax.set_xlabel('time')
        ax.set_ylabel(self.unit)
        ax.grid(True)
        ax.legend()

        # Return an axis
        return ax

    def timeshift(self, offset: float):
        """
        Offset the time base by an arbitrary value

        Parameters
        ----------
        offset : float
            Value by which time will be offset

        Returns
        -------
        Series
            Modified time series

        """

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Return as timedomain with offset
        return TimeDomain._reconstruct(x + offset, y, self._name, self.unit, self._start_time)

    def deduplicate(self):
        """
        Deduplicate repeated values in X

        Note: Use this function at your own risk! It will compute a mean over duplicate values. This may not always be
        a desirable behavior.

        Returns
        -------
        Series
            Modified time series

        """

        return self._obj.groupby(self._obj.index).mean().reset_index(drop=True)

    def between(self, start: float, end: float):
        """
        Get time series between and including two points in time

        Parameters
        ----------
        start : float
            Lower time value
        end : float
            Upper time value

        Returns
        -------
        Series
            Modified time series

        """

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # Call series function
        x2, y2 = _between(x, y, start, end)
        return TimeDomain._reconstruct(x2, y2, self._name, self.unit, self._start_time)

    def playsound(self, wait=False):
        """
        Play a sound from data with a constant logging rate
        """

        # Underlying data
        y = self._get_y()
        fs = self.samplerate

        # Play using sounddevice
        sd.play(y, fs)

        # Wait if asked
        if wait:
            sd.wait()

    def filt_butter(self, cutoff, order: int, btype: str):
        """
        Filter the timedomain using a butterworth filter. Filter operates in
        both directions (filtfilt) to remove phase.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz
        order : int
            The order of the filter
        btype : str
            Type: lowpass, highpass, bandpass, bandstop

        Returns
        -------
        Series
            Modified time series

        """

        # Nyquist frequency
        nyq = 0.5 * self.samplerate

        # Normalize cutoff by the nyquist frequency
        normal_cutoff = cutoff / nyq

        # Filter order
        order_used = order / 2  # divide by 2 since filtfilt doubles the order
        if order_used != math.ceil(order_used):
            order_used = math.ceil(order_used)
            print("Order must be even integer, it was bumped to %d" % (order_used * 2))

        # Construct the filter coefficients
        b, a = scipy.signal.butter(order_used, normal_cutoff, btype=btype, analog=False)

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Filter the data
        y_filt = scipy.signal.filtfilt(b, a, y)
        name = 'filt(' + self._name + ')'

        # Return a new timedomain object
        return TimeDomain._reconstruct(x, y_filt, name, self.unit, self._start_time)

    def filt_cfc(self, cfc):
        """
        Filter the timedomain using a CFC filter per SAE J1211

        Parameters
        ----------
        cfc : float
            Cutoff frequency in Hz

        Returns
        -------
        Series
            Modified time series

        """

        # Coefficients per SAE-J1211
        dt = 1 / self.samplerate
        wd = 2 * math.pi * cfc * 2.0775
        wa = math.sin(wd*dt/2) / math.cos(wd*dt/2)
        a0 = (wa**2) / (1 + math.sqrt(2)*wa + (wa**2))
        a1 = 2 * a0
        a2 = a0
        b0 = 1
        b1 = -2 * (wa**2-1) / (1 + math.sqrt(2)*wa + wa**2)
        b2 = (-1 + math.sqrt(2)*wa - wa**2) / (1 + math.sqrt(2)*wa + wa**2)

        # Build lists
        b = [a0, a1, a2]
        a = [b0, -b1, -b2]

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Filter the data
        y_filt = scipy.signal.filtfilt(b, a, y)
        name = 'CFC%d(%s)' % (cfc, self._name)

        # Return a new timedomain object
        return TimeDomain._reconstruct(x, y_filt, name, self.unit, self._start_time)

    def filt_a(self):
        """
        Filter the timedomain using an a-weighting filter

        Returns
        -------
        Series
            Modified time series

        """

        # Sample rate
        fs = self.samplerate

        # Construct the filter coefficients
        b, a = sp_tools.a_weighting(fs)

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Filter the data
        y_filt = scipy.signal.lfilter(b, a, y)
        name = 'a_filt(' + self._name + ')'

        # Return a new timedomain object
        return TimeDomain._reconstruct(x, y_filt, name, self.unit, self._start_time)

    def get_psd(self, window_length: float = None, overlap: float = 0.5, window_type: str = 'hann',
                summarize: str = 'mean', fatigue_exponent: float = 4.0,
                detrend='constant'):
        """
        Compute a power spectral density from the timedomain. Uses periodogram
        by default, or Welch's estimate if a window is specified. Uses boxcar
        window function by default, or hanning window if a window is specified.
        https://en.wikipedia.org/wiki/Spectral_density

        Parameters
        ----------
        window_length : float, optional
            Window length in units time for each sub-PSD. The default is None.
        overlap : float, optional
            Overlap ratio for hanning windows. The default is 0.5.
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
        freqdomain
            The PSD object

        """

        # Useful data for the PSD
        y = self._get_y()
        samplerate = self.samplerate

        # Zero window means to just use the whole series, otherwise window is
        # in units of time
        if window_length is None:
            f, pxx = scipy.signal.periodogram(y, samplerate)
        else:
            f, pxx = sp_tools.psd(y, samplerate,
                                  window_length=window_length, overlap=overlap, window_type=window_type,
                                  summarize=summarize, fatigue_exponent=fatigue_exponent,
                                  detrend=detrend)

        # Only return positive freq
        pxx = pxx[f > 0]
        f = f[f > 0]

        # Return a new PSD object
        out_data = pd.Series(pxx, f, name=self._name)
        out_data.freqdomain.unit = self.unit + '^2/Hz'
        return out_data

    def get_srs(self, freq: np.array = None, quality_factor: float = 50.):
        """
        Compute a shock response spectrum
        https://en.wikipedia.org/wiki/Shock_response_spectrum

        Parameters
        ----------
        freq : Numpy array, optional
            Frequency vector. The default is the function
            tools.build_freq_array() with its defaults
        quality_factor : float, optional
            Quality factor. The default is 50.

        Returns
        -------
        Series
            Max absolute acceleration response at each frequency step

        """

        # Default freq vector
        if freq is None:
            freq = sp_tools.build_freq_array()

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Compute the SRS
        pos_accel, neg_accel = sp_tools.srs(x, y, freq, quality_factor)
        name = 'srs(' + self._name + ',Q=' + str(quality_factor) + ')'

        # Return a new PSD object
        out_data = pd.Series(np.maximum(pos_accel, neg_accel), freq, name=name)
        out_data.freqdomain.unit = self.unit
        return out_data

    def mov_rms(self, window: float):
        """
        Compute the moving RMS of the signal given a window length in units of
        time. RMS is calculated for each timestep using the window of specified
        length preceding that time step. Steps leading up to the window length
        will be ommitted from the result.

        Parameters
        ----------
        window : float
            Time length of RMS window.

        Returns
        -------
        Series
            Modified time series

        """

        # Compute window size in terms of number of samples
        window_n = round(window * self.samplerate)

        # Get timestamps and samples
        x = self._get_x()

        # Compute moving RMS
        tmp = abs(self._obj) ** 2
        mov_rms = tmp.rolling(window_n).mean() ** 0.5
        mov_rms = mov_rms.to_numpy(copy=True)
        rms_name = 'mov_rms(' + self._name + ',' + str(window) + ')'

        # Throw away NaNs
        mov_rms = mov_rms[window_n - 1:-1]
        x_new = x[window_n - 1:-1] - (window / 2)

        # Return a new timedomain object
        return TimeDomain._reconstruct(x_new, mov_rms, rms_name, self.unit, self._start_time)

    def integral(self):
        """
        Compute the integral using trapezoidal rule

        Returns
        -------
        Series
            Modified time series

        """

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Trapezoidal integration
        y_int = scipy.integrate.cumulative_trapezoid(y, x, initial=0)
        int_name = 'int(' + self._name + ')'

        # Return a new timedomain object
        # To-do: unit calculator
        return TimeDomain._reconstruct(x, y_int, int_name, "none", self._start_time)

    def derivative(self):
        """
        Compute the derivative

        Returns
        -------
        Series
            Modified time series

        """

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Get the derivative
        y_prime = np.diff(y) / np.diff(x)
        der_name = 'diff(' + self._name + ')'

        # New time base
        x_new = (x[:-1] + x[1:]) / 2

        # Return a new timedomain object
        # To-do: unit calculator
        return TimeDomain._reconstruct(x_new, y_prime, der_name, "none", self._start_time)

    def detrend(self, dt_type: str = 'linear', breakpoints=0):
        """
        Compute the derivative

        Parameters
        ----------
        dt_type : {‘linear’, ‘constant’}, optional
            The type of detrending. If type == 'linear' (default), the result of a linear least-squares fit to data is
            subtracted from data. If type == 'constant', only the mean of data is subtracted.
        breakpoints : array_like of ints, optional
            A sequence of break points. If given, an individual linear fit is performed for each part of data between
            two break points. Break points are specified as indices into data. This parameter only has an effect when
            type == 'linear'.

        Returns
        -------
        Series
            Modified time series

        """

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # De-trend the data
        y_det = scipy.signal.detrend(y, -1, dt_type, breakpoints)
        det_name = 'detrend(' + self._name + ')'

        # Return a new timedomain object
        return TimeDomain._reconstruct(x, y_det, det_name, self.unit, self._start_time)

    def rss(self, *args):
        """
        Compute the root-sum-square

        Parameters
        ----------
        **args : timedomain
            One or more timedomain to RSS with

        Returns
        -------
        Series
            Modified time series

        """

        # Initialize output
        rss_name = 'rss(' + self._name
        sum_squared = self._obj ** 2

        # Loop to compute sum squared
        for arg in args:
            sum_squared = sum_squared + arg ** 2
            rss_name = rss_name + ',' + arg.name

        # Finish computation
        rss_name = rss_name + ')'
        root_sum_squared = sum_squared ** (1 / 2)

        # Get timestamps and samples
        x_rss = root_sum_squared.index.to_numpy(copy=True)
        y_rss = root_sum_squared.to_numpy(copy=True)

        # Return a new timedomain object
        return TimeDomain._reconstruct(x_rss, y_rss, rss_name, self.unit, self._start_time)

    def interp1d(self, x2, kind: str = 'linear'):
        """
        Interpolate a 1-D function

        Parameters:
        ----------
        x2: array-like
            The new x values to interpolate against
        kind : str, optional
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the
            previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

        Returns
        -------
        Series
            Modified time series

        """

        # Name
        inp_name = 'interp(' + str(self._obj.name) + ')'

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # Call series function
        y2 = _interp1d(x, y, x2, kind)
        return TimeDomain._reconstruct(x2, y2, inp_name, self.unit, self._start_time)

    def resample(self, sample_rate: float, interp_kind: str = 'linear'):
        """
        Set a constant sample rate for the timedomain

        Parameters:
        ----------
        sample_rate: float
            The new sample rate
        interp_kind : str, optional
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the
            previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

        Returns
        -------
        Series
            Modified time series

        """

        # Get timestamps
        x = self._get_x()

        # Properties of current time base
        start_time = x[0]
        end_time = x[-1]

        # New sample rate
        dt = 1 / sample_rate

        # New time vector
        x_new = np.arange(start_time, end_time, dt)

        # Interpolate to create new timedomain
        new_timedomain = self.interp1d(x_new, interp_kind)
        new_timedomain.rename(self._name, inplace=True)

        # Return timedomain
        return new_timedomain

    def rms(self):
        """
        Get the root-mean-square of the timedomain

        Returns
        -------
        float
            RMS value

        """

        # Get samples
        y = self._get_y()

        # Return RMS
        return sp_tools.rms(y)

    def spl(self, p_ref=20E-6):
        """
        Get the sound pressure level of the timedomain

        Parameters
        ----------
        p_ref
            Reference pressure, defaults to 20*10^-6, which
            is the default for air in pascals

        Returns
        -------
        float
            RMS value

        """

        # Get samples
        y = self._get_y()

        # Return RMS
        return sp_tools.spl(y, p_ref)

    def rainflow(self, **kwargs):
        """
        Python implementation of the ASTM E1049-85 rainflow cycle counting algorythm for fatigue analysis. Supports
        both Python 2 and 3.

        Parameters
        ----------
        ndigits
            Round cycle magnitudes to the given number of digits before counting. Use a negative value to round to
            tens, hundreds, etc.
        nbins
            Specifies the number of cycle-counting bins.
        binsize
            Specifies the width of each cycle-counting bin

        Returns
        -------
        float
            bin, cycles

        """

        # Get samples
        y = self._get_y()

        return rf.count_cycles(y, **kwargs)

    @staticmethod
    def _reconstruct(x, y, name, unit, origin):
        """Reconstruct a timedomain"""

        out_data = pd.Series(y, x, name=name)
        if origin is not None:
            out_data.index = pd.to_datetime(out_data.index, unit="s", origin=origin)
        out_data.timedomain.unit = unit
        return out_data


@pd.api.extensions.register_series_accessor("freqdomain")
class FreqDomain:
    """
    Container and functions for working with power spectral density
    """

    _obj = None
    "Object property"

    unit = "none"
    "Engineering units"

    def __init__(self, pd_series: pd.Series):
        """
        Class constructor

        Parameters
        ----------
        pd_series : pandas Series
            Pandas series object

        """

        # Construct in the superclass
        pd_series.index.name = "time"
        self._obj = pd_series

    def _get_x(self):
        """Get the x values"""
        return self._obj.index.to_numpy(copy=True)

    def _get_y(self):
        """Get the y values"""
        return self._obj.to_numpy(copy=True)

    @property
    def _name(self):
        """Name of the series"""
        return str(self._obj.name)

    def plot(self, ax: plt.axis = None, loglog: bool = True, **kwargs):
        """
        Plot the freqdomain

        Parameters
        ----------
        ax : pyplot axis, optional
            Pass an existing axis on which to draw the plot. The default is
            None.
        loglog : bool, default True
            Plot in loglog space
        **kwargs
            optional arguments for pyplot

        Returns
        -------
        ax : pyplot axis
            The axis used for this plot

        """

        # Create a new figure and axis if one isn't supplied
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        # Basic plotting
        self._obj.plot(ax=ax,
                       xlabel='frequency',
                       ylabel=self.unit,
                       grid=True,
                       legend=True,
                       **kwargs)

        # PSD best plotted in log-log by default
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # Return an axis
        return ax

    def between(self, start: float, end: float):
        """
        Get PSD between and including two points in freq domain

        Parameters
        ----------
        start : float
            Lower frequency value
        end : float
            Upper frequency value

        Returns
        -------
        Series
            Modified freq series

        """

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # Call series function
        x2, y2 = _between(x, y, start, end)
        return FreqDomain._reconstruct(x2, y2, self._name, self.unit)

    def rms_lin(self):
        """
        Get the RMS of the PSD in linear space. This is suitable for PSDs which
        are densely populated, such as PSDs calculated from a time history.

        Returns
        -------
        float
            The RMS value

        """

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # RMS of PSD in log-log space
        return sp_tools.rms_psd_linear(y, x)

    def rms_log(self):
        """
        Get the RMS of the PSD in log-log space, recommended for sparsely
        populated PSDs which have linear regions on a log-log plot. Typically
        specs fall into this category. Densely populated PSDs will likely run
        out of machine precision and need to revert to linear space for the
        integral.

        Returns
        -------
        float
            The RMS value

        """

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # RMS of PSD in log-log space
        return sp_tools.rms_psd_loglog(y, x)

    def interp1d(self, x2, kind: str = 'linear'):
        """
        Interpolate a 1-D function

        Parameters:
        ----------
        x2: array-like
            The new x values to interpolate against
        kind : str, optional
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the
            previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

        Returns
        -------
        Series
            Modified freq series

        """

        # Name
        inp_name = 'interp(' + str(self._obj.name) + ')'

        # Underlying data
        x = self._get_x()
        y = self._get_y()

        # Call series function
        y2 = _interp1d(x, y, x2, kind)
        return FreqDomain._reconstruct(x2, y2, inp_name, self.unit)

    def interp1d_log(self, x2):
        """
        Interpolate a 1-D function in log-log space

        Parameters:
        ----------
        x2: array-like
            The new x values to interpolate against
        kind : str or int, optional
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the
            previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

        Returns
        -------
        Series
            Modified freq series

        """

        # Get timestamps and samples
        x = self._get_x()
        y = self._get_y()

        # Create interpolation function
        f = scipy.interpolate.interp1d(np.log10(x), np.log10(y))
        name = 'interp(' + self._name + ')'

        # Return val
        y2 = f(np.log10(x2))
        y2 = 10 ** y2

        # Return as PSD
        return FreqDomain._reconstruct(x2, y2, name, self.unit)

    @staticmethod
    def _reconstruct(x, y, name, unit):
        """Reconstruct a timedomain"""

        out_data = pd.Series(y, x, name=name)
        out_data.freqdomain.unit = unit
        return out_data


def _between(x: np.array, y: np.array, start: float, end: float):
    """
    Get between and including two points
    """

    # Select subset
    y2 = y[(x <= end) & (x >= start)]
    x2 = x[(x <= end) & (x >= start)]

    return x2, y2


def _interp1d(x: np.array, y: np.array, x2, kind: str = 'linear'):
    """
    Interpolate a pandas series
    """

    # Create interpolation function
    f = scipy.interpolate.interp1d(x, y, kind)

    # Return val
    y2 = f(x2)
    return y2
