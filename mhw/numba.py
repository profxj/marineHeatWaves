""" Routines sped up by numba"""
from numba import njit, prange
import numpy as np

'''
@njit(parallel=True)
def event_stats(n_events, t, mhw, temp, thresh, seas):
    for ev in prange(n_events):
        # Get SST series during MHW event, relative to both threshold and to seasonal climatology
        tt_start = np.where(t == mhw['time_start'][0][ev])[0][0]
        tt_end = np.where(t == mhw['time_end'][0][ev])[0][0]
        # mhw['date_start'].append(date.fromordinal(mhw['time_start'][ev]))
        # mhw['date_end'].append(date.fromordinal(mhw['time_end'][ev]))
        # mhw['index_start'].append(tt_start)
        # mhw['index_end'].append(tt_end)
        temp_mhw = temp[tt_start:tt_end + 1]
        thresh_mhw = thresh[tt_start:tt_end + 1]
        seas_mhw = seas[tt_start:tt_end + 1]
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relThresh = temp_mhw - thresh_mhw
        mhw_relThreshNorm = (temp_mhw - thresh_mhw) / (thresh_mhw - seas_mhw)
        mhw_abs = temp_mhw

        # Fill
        # Find peak
        tt_peak = np.argmax(mhw_relSeas)
        mhw['time_peak'][0][ev] = mhw['time_start'][0][ev] + tt_peak
        #mhw['date_peak'].append(date.fromordinal(mhw['time_start'][ev] + tt_peak))
        #mhw['index_peak'].append(tt_start + tt_peak)
'''


@njit(parallel=True)
def event_stats(n_events, t, temp, thresh, seas, time_start, time_end,
                    time_peak, duration, intensity_max, intensity_mean, intensity_var,
                    intensity_cumulative, intensity_max_relThresh, intensity_mean_relThresh,
                    intensity_var_relThresh, intensity_cumulative_relThresh, intensity_max_abs,
                    intensity_mean_abs, intensity_var_abs, intensity_cumulative_abs, category,
                    duration_moderate, duration_strong, duration_severe, duration_extreme,
                    rate_onset, rate_decline):
    """

    Parameters
    ----------
    n_events
    t
    temp
    thresh
    seas
    time_start
    time_end
    time_peak
    duration
    intensity_max
    intensity_mean
    intensity_var
    intensity_cumulative
    intensity_max_relThresh
    intensity_mean_relThresh
    intensity_var_relThresh
    intensity_cumulative_relThresh
    intensity_max_abs
    intensity_mean_abs
    intensity_var_abs
    intensity_cumulative_abs
    category
    duration_moderate
    duration_strong
    duration_severe
    duration_extreme
    rate_onset
    rate_decline

    """
    #for ev in range(n_events):
    for ev in prange(n_events):
        # Get SST series during MHW event, relative to both threshold and to seasonal climatology
        tt_start = np.where(t == time_start[ev])[0][0]
        tt_end = np.where(t == time_end[ev])[0][0]
        # mhw['date_start'].append(date.fromordinal(mhw['time_start'][ev]))
        # mhw['date_end'].append(date.fromordinal(mhw['time_end'][ev]))
        # mhw['index_start'].append(tt_start)
        # mhw['index_end'].append(tt_end)
        temp_mhw = temp[tt_start:tt_end + 1]
        thresh_mhw = thresh[tt_start:tt_end + 1]
        seas_mhw = seas[tt_start:tt_end + 1]
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relThresh = temp_mhw - thresh_mhw
        mhw_relThreshNorm = (temp_mhw - thresh_mhw) / (thresh_mhw - seas_mhw)
        mhw_abs = temp_mhw

        # Fill
        # Find peak
        tt_peak = np.argmax(mhw_relSeas)
        time_peak[ev] = time_start[ev] + tt_peak
        #mhw['date_peak'].append(date.fromordinal(mhw['time_start'][ev] + tt_peak))
        #mhw['index_peak'].append(tt_start + tt_peak)
        # MHW Duration
        duration[ev] = len(mhw_relSeas)
        # MHW Intensity metrics
        intensity_max[ev] = mhw_relSeas[tt_peak]
        intensity_mean[ev] = mhw_relSeas.mean()
        intensity_var[ev] = np.sqrt(mhw_relSeas.var())
        intensity_cumulative[ev] = mhw_relSeas.sum()
        intensity_max_relThresh[ev] = mhw_relThresh[tt_peak]
        intensity_mean_relThresh[ev] = mhw_relThresh.mean()
        intensity_var_relThresh[ev] = np.sqrt(mhw_relThresh.var())
        intensity_cumulative_relThresh[ev] = mhw_relThresh.sum()
        intensity_max_abs[ev] = mhw_abs[tt_peak]
        intensity_mean_abs[ev] = mhw_abs.mean()
        intensity_var_abs[ev] = np.sqrt(mhw_abs.var())
        intensity_cumulative_abs[ev] = mhw_abs.sum()
        # Fix categories
        tt_peakCat = np.argmax(mhw_relThreshNorm)
        cats = np.floor(1. + mhw_relThreshNorm)
        pcat = int(cats[tt_peakCat])
        #mhw['category'][ev] = categories[np.min([cats[tt_peakCat], 4]).astype(int) - 1]
        category[ev] = min([pcat, 4]) - 1
        duration_moderate[ev] = int(np.sum(cats == 1))
        duration_strong[ev] = int(np.sum(cats == 2))
        duration_severe[ev] = int(np.sum(cats == 3))
        duration_extreme[ev] = int(np.sum(cats >= 4))

        # Rates of onset and decline
        # Requires getting MHW strength at "start" and "end" of event (continuous: assume start/end half-day before/after first/last point)
        if tt_start > 0:
            mhw_relSeas_start = 0.5*(mhw_relSeas[0] + temp[tt_start-1] - seas[tt_start-1])
            rate_onset[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas_start) / (tt_peak+0.5)
        else: # MHW starts at beginning of time series
            if tt_peak == 0: # Peak is also at begining of time series, assume onset time = 1 day
                rate_onset[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas[0]) / 1.
            else:
                rate_onset[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas[0]) / tt_peak
        if tt_end < len(t)-1:
            mhw_relSeas_end = 0.5*(mhw_relSeas[-1] + temp[tt_end+1] - seas[tt_end+1])
            rate_decline[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas_end) / (tt_end-tt_start-tt_peak+0.5)
        else: # MHW finishes at end of time series
            if tt_peak == len(t)-1: # Peak is also at end of time series, assume decline time = 1 day
                rate_decline[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / 1.
            else:
                rate_decline[ev] = (mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / (tt_end-tt_start-tt_peak)


