from numba import njit, prange
import numpy as np

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
        # MHW Duration
        mhw['duration'][ev] = len(mhw_relSeas)
        # MHW Intensity metrics
        mhw['intensity_max'][ev] = mhw_relSeas[tt_peak]
        mhw['intensity_mean'][ev] = mhw_relSeas.mean()
        mhw['intensity_var'][ev] = np.sqrt(mhw_relSeas.var())
        mhw['intensity_cumulative'][ev] = mhw_relSeas.sum()
        mhw['intensity_max_relThresh'][ev] = mhw_relThresh[tt_peak]
        mhw['intensity_mean_relThresh'][ev] = mhw_relThresh.mean()
        mhw['intensity_var_relThresh'][ev] = np.sqrt(mhw_relThresh.var())
        mhw['intensity_cumulative_relThresh'][ev] = mhw_relThresh.sum()
        mhw['intensity_max_abs'][ev] = mhw_abs[tt_peak]
        mhw['intensity_mean_abs'][ev] = mhw_abs.mean()
        mhw['intensity_var_abs'][ev] = np.sqrt(mhw_abs.var())
        mhw['intensity_cumulative_abs'][ev] = mhw_abs.sum()
        # Fix categories
        tt_peakCat = np.argmax(mhw_relThreshNorm)
        cats = np.floor(1. + mhw_relThreshNorm)
        #mhw['category'][ev] = categories[np.min([cats[tt_peakCat], 4]).astype(int) - 1]
        mhw['category'][ev] = np.min([cats[tt_peakCat], 4]).astype(int) - 1
        mhw['duration_moderate'][ev] = np.sum(cats == 1.)
        mhw['duration_strong'][ev] = np.sum(cats == 2.)
        mhw['duration_severe'][ev] = np.sum(cats == 3.)
        mhw['duration_extreme'][ev] = np.sum(cats >= 4.)
        '''


@njit(parallel=True)
def dev_event_stats(n_events, t, temp, thresh, seas, time_start, time_end, time_peak):
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
        '''
        # MHW Duration
        mhw['duration'][ev] = len(mhw_relSeas)
        # MHW Intensity metrics
        mhw['intensity_max'][ev] = mhw_relSeas[tt_peak]
        mhw['intensity_mean'][ev] = mhw_relSeas.mean()
        mhw['intensity_var'][ev] = np.sqrt(mhw_relSeas.var())
        mhw['intensity_cumulative'][ev] = mhw_relSeas.sum()
        mhw['intensity_max_relThresh'][ev] = mhw_relThresh[tt_peak]
        mhw['intensity_mean_relThresh'][ev] = mhw_relThresh.mean()
        mhw['intensity_var_relThresh'][ev] = np.sqrt(mhw_relThresh.var())
        mhw['intensity_cumulative_relThresh'][ev] = mhw_relThresh.sum()
        mhw['intensity_max_abs'][ev] = mhw_abs[tt_peak]
        mhw['intensity_mean_abs'][ev] = mhw_abs.mean()
        mhw['intensity_var_abs'][ev] = np.sqrt(mhw_abs.var())
        mhw['intensity_cumulative_abs'][ev] = mhw_abs.sum()
        # Fix categories
        tt_peakCat = np.argmax(mhw_relThreshNorm)
        cats = np.floor(1. + mhw_relThreshNorm)
        #mhw['category'][ev] = categories[np.min([cats[tt_peakCat], 4]).astype(int) - 1]
        mhw['category'][ev] = np.min([cats[tt_peakCat], 4]).astype(int) - 1
        mhw['duration_moderate'][ev] = np.sum(cats == 1.)
        mhw['duration_strong'][ev] = np.sum(cats == 2.)
        mhw['duration_severe'][ev] = np.sum(cats == 3.)
        mhw['duration_extreme'][ev] = np.sum(cats >= 4.)
        '''


