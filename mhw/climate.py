'''

    A set of functions which implement the Marine Heat Wave (MHW)
    definition of Hobday et al. (2016)

'''


import numpy as np
from datetime import date

from numba import njit, prange

from mhw import utils

from IPython import embed

def build_time_dict(t):

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
    year_leapYear = 2012 # This year was a leap-year and therefore doy in range of 1 to 366
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(),date(year_leapYear, 12, 31).toordinal()+1)
    #dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year,1,1).toordinal() + 1
    # Calculate day-of-year values
    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]

    times = {}
    times['t'] = t
    times['year'] = year.astype(int)
    times['doy'] = doy.astype(int)

    return times

def calc(times, temp, windowHalfWidth=5, maxPadLength=False, Ly=False,
         smoothPercentile=True, smoothPercentileWidth=31, pctile=90,
         parallel=False):
    '''

    Applies the Hobday et al. (2016) marine heat wave definition to an input time
    series of temp ('temp') along with a time vector ('t'). Outputs properties of
    all detected marine heat waves.

    Inputs:

      times   Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
              [1D numpy array of length T of type int]
      temp    Temperature vector [1D numpy array of length T]

    Outputs:

      clim    Climatology of SST. Each key (following list) is a seasonally-varying
              time series [1D numpy array of length T] of a particular measure:

        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle
        'missing'              A vector of TRUE/FALSE indicating which elements in 
                               temp were missing values for the MHWs detection

    Options:

      pctile                 Threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      smoothPercentile       Boolean switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days])
      windowHalfWidth        Width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      maxPadLength           Specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN. Set as an integer.
                             (DEFAULT = False, interpolates over all missing values).
      Ly                     Specifies if the length of the year is < 365/366 days (e.g. a
                             360 day year from a climate model). This affects the calculation
                             of the climatology. (DEFAULT = False)

    Notes:

      1. This function assumes that the input time series consist of continuous daily values
         with few missing values. Time ranges which start and end part-way through the calendar
         year are supported.

      2. This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.

      3. The calculation of onset and decline rates assumes that the heat wave started a half-day
         before the start day and ended a half-day after the end-day. (This is consistent with the
         duration definition as implemented, which assumes duration = end day - start day + 1.)

      4. For the purposes of MHW detection, any missing temp values not interpolated over (through
         optional maxPadLLength) will be set equal to the seasonal climatology. This means they will
         trigger the end/start of any adjacent temp values which satisfy the MHW criteria.

      5. If the code is used to detect cold events (coldSpells = True), then it works just as for heat
         waves except that events are detected as deviations below the (100 - pctile)th percentile
         (e.g., the 10th instead of 90th) for at least 5 days. Intensities are reported as negative
         values and represent the temperature anomaly below climatology.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb 2015

    '''
    import pdb; pdb.set_trace()  #  I BROKE THIS UP AND USE MY OWN HOME-BREW

    #
    # Time and dates vectors
    #

    # Constants (doy values for Feb-28 and Feb-29) for handling leap-years
    feb29 = 60

    #
    # Calculate threshold and seasonal climatology (varying with day-of-year)
    #
    tempClim = temp
    doyClim = times['doy']
    TClim = len(doyClim)

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        temp = utils.pad(temp, maxPadLength=maxPadLength)
        tempClim = utils.pad(tempClim, maxPadLength=maxPadLength)

    # Length of climatological year
    lenClimYear = 366
    # Start and end indices
    #clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
    #clim_end = np.where(yearClim == climatologyPeriod[1])[0][-1]
    clim_start = 0
    clim_end = len(doyClim)

    # Inialize arrays
    thresh_climYear = np.NaN*np.zeros(lenClimYear, dtype='float32')
    seas_climYear = np.NaN*np.zeros(lenClimYear, dtype='float32')
    clim = {}
    #clim['thresh'] = np.NaN*np.zeros(TClim)
    #clim['seas'] = np.NaN*np.zeros(TClim)
    wHW_array = np.outer(np.ones(1000, dtype='int'), np.arange(-windowHalfWidth, windowHalfWidth+1))
    nwHW = wHW_array.shape[1]

    # Loop over all day-of-year values, and calculate threshold and seasonal climatology across years
    if parallel:
        doit(lenClimYear, feb29, doyClim, clim_start, clim_end, wHW_array, nwHW,
                 TClim, thresh_climYear, tempClim, pctile, seas_climYear)
    else:
        for d in range(1,lenClimYear+1):
            # Special case for Feb 29
            if d == feb29:
                continue
            # find all indices for each day of the year +/- windowHalfWidth and from them calculate the threshold
            tt0 = np.where(doyClim[clim_start:clim_end+1] == d)[0]
            # If this doy value does not exist (i.e. in 360-day calendars) then skip it
            if len(tt0) == 0:
                continue
            #tt = np.array([])
            #for w in range(-windowHalfWidth, windowHalfWidth+1):
            #    tt = np.append(tt, clim_start+tt0 + w)
            tt = (wHW_array[0:len(tt0),:] + np.outer(tt0, np.ones(nwHW, dtype='int'))).flatten()
            gd = np.all([tt >= 0, tt<TClim], axis=0)
            tt = tt[gd] # Reject indices "after" the last element
            thresh_climYear[d-1] = np.nanpercentile(tempClim[tt], pctile)
            seas_climYear[d-1] = np.nanmean(tempClim[tt])
    # Special case for Feb 29
    thresh_climYear[feb29-1] = 0.5*thresh_climYear[feb29-2] + 0.5*thresh_climYear[feb29]
    seas_climYear[feb29-1] = 0.5*seas_climYear[feb29-2] + 0.5*seas_climYear[feb29]

    # Smooth if desired
    if smoothPercentile:
        # If the length of year is < 365/366 (e.g. a 360 day year from a Climate Model)
        if Ly:
            valid = ~np.isnan(thresh_climYear)
            thresh_climYear[valid] = utils.runavg(thresh_climYear[valid], smoothPercentileWidth)
            valid = ~np.isnan(seas_climYear)
            seas_climYear[valid] = utils.runavg(seas_climYear[valid], smoothPercentileWidth)
        # >= 365-day year
        else:
            thresh_climYear = utils.runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = utils.runavg(seas_climYear, smoothPercentileWidth)

    # Generate threshold for full time series
    clim['thresh'] = thresh_climYear.astype('float32')#[doyClim-1]
    clim['seas'] = seas_climYear.astype('float32')#[doyClim-1]
    # Save vector indicating which points in temp are missing values
    #clim['missing'] = np.isnan(temp)

    return clim

#@njit(parallel=True)
def doit(lenClimYear, feb29, doyClim, clim_start, clim_end, wHW_array, nwHW,
         TClim, thresh_climYear, tempClim, pctile, seas_climYear):

    ones = np.array([1]*nwHW) #np.ones(nwHW).astype(int)
    for d in range(1,lenClimYear+1):
    #for d in prange(1,lenClimYear+1):
        # Special case for Feb 29
        if d == feb29:
            continue
        # find all indices for each day of the year +/- windowHalfWidth and from them calculate the threshold
        tt0 = np.where(doyClim[clim_start:clim_end+1] == d)[0]
        # If this doy value does not exist (i.e. in 360-day calendars) then skip it
        if len(tt0) == 0:
            continue
        #tt = np.array([])
        #for w in range(-windowHalfWidth, windowHalfWidth+1):
        #    tt = np.append(tt, clim_start+tt0 + w)
        tt = (wHW_array[0:len(tt0),:] + np.outer(tt0, ones)).flatten() #np.ones(nwHW, dtype='int32')))#.flatten()
        #gd = np.all([tt >= 0, tt<TClim], axis=0)
        gd = (tt >= 0) & (tt<TClim)
        tt = tt[gd] # Reject indices "after" the last element
        thresh_climYear[d-1] = np.nanpercentile(tempClim[tt], pctile)
        if d == 353:
            import pdb; pdb.set_trace()
        seas_climYear[d-1] = np.nanmean(tempClim[tt])
