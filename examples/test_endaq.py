# -*- coding: utf-8 -*-

# Imports
import daq.io as daq_io

# %% User Entry

# Path to the log file
filepath = r'<file path here>'

# Channel to plot (these names depend on SlamStick model)
channel_nameX = 'Acceleration: X (100g)'
channel_nameY = 'Acceleration: Y (100g)'
channel_nameZ = 'Acceleration: Z (100g)'

# CFC Filter Cutoff (1000 Hz for head accelerations per SAE J-211)
f_cfc = 1000

# %% Code

# Link to DAQ
my_daq = daq_io.Endaq(filepath)

# Get channels
my_channel_X = my_daq.get_channel(channel_nameX)
my_channel_Y = my_daq.get_channel(channel_nameY)
my_channel_Z = my_daq.get_channel(channel_nameZ)

# Convert to series type
my_series_X = my_channel_X.to_pandas()
my_series_Y = my_channel_Y.to_pandas()
my_series_Z = my_channel_Z.to_pandas()

# Compute CFC filter
my_filt_X = my_series_X.timedomain.filt_cfc(f_cfc)
my_filt_Y = my_series_Y.timedomain.filt_cfc(f_cfc)
my_filt_Z = my_series_Z.timedomain.filt_cfc(f_cfc)

# Plot the timeseries
ax1 = my_series_X.timedomain.plot()
my_filt_X.timedomain.plot(ax1)
my_filt_Y.timedomain.plot(ax1)
my_filt_Z.timedomain.plot(ax1)
