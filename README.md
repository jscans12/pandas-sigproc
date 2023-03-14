# Pandas Signal Processing (pandas-sigproc)
Useful pandas extensions for detailed signal processing. 

This package contains extensions for the [pandas](https://pandas.pydata.org/) library. It is a Python package and can be installed via pip:

```python
pip install pandas-sigproc
```

Once installed, the package can be referenced as such:

```python
import pandas_sigproc.extension as pd_sigproc
```

There are two extensions included in the package:
- freqdomain: analyze pandas series in the frequency domain
- timedomain: analyze pandas series in the time domain

The code should be referenced for a more thorough understanding of what functions are available. However, given a pandas (time) series called my_series which has index time and values sound pressure, here is an example of how one would apply an a-weighted filter to that data:

```python
my_series_aweighted = my_series.timedomain.filt_a()
```
