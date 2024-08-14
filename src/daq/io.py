# -*- coding: utf-8 -*-
"""
Methods for handling log files
@author: john.scanlon
"""

from abc import ABC, abstractmethod
import dwdatareader as dw
import endaq.ide as ide
import pandas as pd
import pandas_sigproc.extension


class BaseFile(ABC):
    """Abstract class for interface to log file"""

    @property
    @abstractmethod
    def filename(self):
        """Path to the logged data file"""
        pass

    @property
    @abstractmethod
    def channel_list(self):
        """Pass a list of channels"""
        pass

    @abstractmethod
    def get_channel(self, ch_id):
        """Pass a channel object"""
        pass


class BaseChannel(ABC):
    """Abstract class for interface to channel"""

    @property
    @abstractmethod
    def start_time(self):
        """Pass the start time"""
        pass

    @property
    @abstractmethod
    def time(self):
        """Pass the time"""
        pass

    @property
    @abstractmethod
    def data(self):
        """Pass the data"""
        pass

    @property
    @abstractmethod
    def name(self):
        """Pass the channel name"""
        pass

    @property
    @abstractmethod
    def unit(self):
        """Pass the channel units"""
        pass

    def to_pandas(self):
        """
        Get timestamps and values from the logged channel as a pandas series

        Returns
        -------
        series
            Pandas series object

        """

        series_obj = pd.Series(data=self.data, index=self.time, name=self.name, copy=True)
        series_obj.index = pd.to_datetime(series_obj.index, unit="s", origin=self.start_time)
        series_obj.timedomain.unit = self.unit
        return series_obj


class Channel(BaseChannel):
    """
    A generic channel
    """

    _start_time = None
    """Logging start time"""

    _time = None
    """Timebase"""

    _data = None
    """Data values"""

    _name = None
    """Channel name"""

    _unit = None
    """Channel units"""

    @property
    def start_time(self):
        """Returns the start time of this channel"""

        return self._start_time

    @property
    def time(self):
        """Returns the time vector of this channel"""

        return self._time

    @property
    def data(self):
        """Returns the data vector of this channel"""

        return self._data

    @property
    def name(self):
        """Returns the name of this channel"""

        return self._name

    @property
    def unit(self):
        """Returns the engineering units of this channel"""

        return self._unit

    def __init__(self, start_time, time, data, name, unit):
        """
        Class constructor

        Parameters
        ----------
        start_time : datetime
            Data start
        time : float
            Array of timestamps in seconds
        data : float
            Data array
        name : string
            Name of the channel
        unit : string
            Engineering units of the channel

        """

        self._start_time = start_time
        self._time = time
        self._data = data
        self._name = name
        self._unit = unit


class Endaq(BaseFile):
    """
    Open and read data from Endaq log files (.ide)
    """

    __fileLink = None
    "Links to the Endaq object upon instantiation"

    @property
    def filename(self):
        """Return the file name"""

        return self.__fileLink.filename

    @property
    def channel_list(self):
        """Return a list of channel names"""

        channel_list = list()
        for i_channel in self.__fileLink.channels.values():
            for j_channel in i_channel.subchannels:
                channel_list.append(j_channel.displayName)

        return channel_list

    def __init__(self, filename: str):
        """
        Class constructor

        Parameters
        ----------
        filename : str
            Full path to the log file

        """

        # Link to the file
        self.__fileLink = ide.get_doc(filename=filename)

    def __del__(self):
        """
        Class destructor
        """

        # Close data file
        self.__fileLink.close()

    def get_channel(self, ch_name):
        """
        Get a channel either by name or index

        Parameters
        ----------
        ch_name : str
            Name of the channel

        Returns
        -------
        EndaqChannel
            The channel object

        """

        # Verify that the channel exists
        if ch_name not in self.channel_list:
            raise Exception("Channel does not exist")

        # Loop to find the channel
        ch_obj = None
        for i_channel in self.__fileLink.channels.values():
            for j_channel in i_channel.subchannels:
                if j_channel.displayName == ch_name:
                    ch_obj = j_channel

        # Raise error if not found
        if ch_obj is None:
            raise Exception("Channel was not found")

        return EndaqChannel(ch_obj)


class EndaqChannel(BaseChannel):
    """
    An Endaq logged channel
    """

    __ch_obj = None
    "Internal channel object"

    @property
    def start_time(self):
        """Returns the start time of this channel"""

        pd_obj = ide.to_pandas(self.__ch_obj, time_mode="datetime", tz="utc")
        return pd_obj.index.min().replace(tzinfo=None)

    @property
    def time(self):
        """Returns the time vector of this channel"""

        pd_obj = ide.to_pandas(self.__ch_obj, time_mode="seconds")
        return pd_obj.index

    @property
    def data(self):
        """Returns the data vector of this channel"""

        pd_obj = ide.to_pandas(self.__ch_obj).squeeze()
        return pd_obj.to_numpy()

    @property
    def name(self):
        """Returns the name of this channel"""

        return self.__ch_obj.displayName

    @property
    def unit(self):
        """Returns the engineering units of this channel"""

        return self.__ch_obj.units[1]

    def __init__(self, ch_obj):
        """
        Class constructor

        Parameters
        ----------
        ch_obj : Endaq channel
            Link to the Dewesoft channel

        """

        self.__ch_obj = ch_obj

    def __to_pandas(self):
        """
        Standard method for pandas conversion
        """

        return ide.to_pandas(self.__ch_obj, time_mode="datetime", tz="utc").squeeze()
