""" Utilities for marine heat waves"""
import glob, os

from IPython.terminal.embed import embed
import numpy as np
import scipy.ndimage as ndimage
from datetime import date

import xarray

def calc_doy(t):
    # Generate vectors for year, month, day-of-month, and day-of-year
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    doy = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day
    # Leap-year baseline for defining day-of-year values
    year_leapYear = 2012  # This year was a leap-year and therefore doy in range of 1 to 366
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(), date(year_leapYear, 12, 31).toordinal() + 1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year, 1, 1).toordinal() + 1
    # Calculate day-of-year values
    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]
    # Return
    return doy.astype(int)


def runavg(ts, w):
    '''

    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:

      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window

    Outputs:

      ts_smooth     Smoothed time series

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts


def pad(data, maxPadLength=False):
    '''

    Linearly interpolate over missing data (NaNs) in a time series.

    Inputs:

      data	     Time series [1D numpy array]
      maxPadLength   Specifies the maximum length over which to interpolate,
                     i.e., any consecutive blocks of NaNs with length greater
                     than maxPadLength will be left as NaN. Set as an integer.
                     maxPadLength=False (default) interpolates over all NaNs.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Jun 2015

    '''
    data_padded = data.copy()
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data_padded[bad_indexes] = interpolated
    if maxPadLength:
        blocks, n_blocks = ndimage.label(np.isnan(data))
        for bl in range(1, n_blocks+1):
            if (blocks==bl).sum() > maxPadLength:
                data_padded[blocks==bl] = np.nan

    return data_padded


def nonans(array):
    '''
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]


def load_noaa_sst(noaa_path:str, sst_root:str,
                 climatologyPeriod:tuple, 
                 interpolated=False):
    """

    Args:
        noaa_path (str):
        sst_root (str):
        climatologyPeriod (tuple):
        interpolated (bool, optional):
            Interpolated SST files

    Returns:
        tuple:  lat_coord, lon_coord, np.array of toordials, list of masked SST

    """
    # Grab the list of SST V1 files
    all_sst_files = glob.glob(os.path.join(noaa_path, sst_root))
    all_sst_files.sort()

    # Load the Cubes into memory
    for ii, ifile in enumerate(all_sst_files):
        if str(climatologyPeriod[0]) in ifile:
            istart = ii
        if str(climatologyPeriod[-1]) in ifile:
            iend = ii+1
    sst_files = all_sst_files[istart:iend]

    print("Loading up the files. Be patient...")
    all_sst = []
    allts = []
    for kk, ifile in enumerate(sst_files):
        print(ifile)  # For progress
        ds = xarray.open_dataset(ifile)
        # lat, lon
        if kk == 0:
            lat = ds.lat 
            lon = ds.lon
        # Allow for interpolated files
        if interpolated:
            t = ds.time.data.astype(int).tolist()
            sst = ds.interpolated_sst.astype('float32').to_masked_array()
        else:
            datetimes = ds.time.values.astype('datetime64[s]').tolist()
            t = [datetime.toordinal() for datetime in datetimes]
            sst = ds.sst.to_masked_array()
        # Append 
        all_sst.append(sst)
        allts += t
    #
    return lat, lon, np.array(allts), all_sst


    

'''
def grab_t(ds_list):
    """
    Grab the times

    Parameters
    ----------
    ds_list (list): List of xarray DataSets

    Returns
    -------
    allts : numpy.ndarray of toordinals (int)

    """

    allts = []
    # For iris
    #for sst in sst_list:
    #    allts += (sst.coord('time').points + 657072).astype(int).tolist()  # 1880?
    for ds in ds_list:
        datetimes = ds.time.values.astype('datetime64[s]').tolist()
        t = [datetime.toordinal() for datetime in datetimes]
        allts += t
    return np.array(allts)
'''

def grab_T(sst_list, i, j):
    """
    Grab an array of SST values

    Parameters
    ----------
    sst_list : list of masked np.array
    i : int
    j : int

    Returns
    -------
    allTs : numpy.ndarray of SST values with nan

    """
    allTs = []
    for sst in sst_list:
        allTs += [sst[:,i,j]]
    return np.ma.concatenate(allTs)
