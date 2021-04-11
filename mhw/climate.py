""" Climate calculations.
"""

import os
import glob
from IPython.core.magic_arguments import MagicHelpFormatter
from pkg_resources import resource_filename

import numpy as np
from scipy import signal

from datetime import date
import datetime

import pandas
import xarray
#import iris

from mhw import utils as mhw_utils
from mhw import mhw_numba

from IPython import embed


def noaa_seas_thresh(climate_db_file,
                     noaa_path=None,
                     climatologyPeriod=(1983, 2012),
                     cut_sky=True, all_sst=None,
                     scale_file=None,
                     pctile=90.,
                     min_frac=0.9, n_calc=None, debug=False):
    """
    Build climate model for NOAA OI data

    Parameters
    ----------
    climate_db_file : str
        output filename.  Should have extension .nc
    noaa_path : str, optional
        Path to NOAA OI SST files
        Defults to os.getenv("NOAA_OI")
    climatologyPeriod : tuple, optional
        Range of years defining the climatology; inclusive
    cut_sky : bool, optional
        Used for debugging
    all_sst : list, optional
        Loaded SST data. For debugging
    pctile : float, optional
        Percentile for T threshold
    min_frac : float
        Minimum fraction required for analysis
    scale_file : str, optional
        Used to remove the overall trend in SST warming
    n_calc : int, optional
        Used for debugging
    debug : bool, optional
        Turn on debugging
    """
    # Path
    if noaa_path is None:
        noaa_path = os.getenv("NOAA_OI")

    # Grab the list of SST V2 files
    all_sst_files = glob.glob(os.path.join(noaa_path, 'sst*nc'))
    all_sst_files.sort()
    # Cut on years
    if '1981' not in all_sst_files[0]:
        raise ValueError("Years not in sync!!")

    # Load the Cubes into memory
    if all_sst is None:
        istart = climatologyPeriod[0] - 1981
        iend = climatologyPeriod[1] - 1981 + 1
        all_sst_files = all_sst_files[istart:iend]

        print("Loading up the files. Be patient...")
        lat_coord, lon_coord, t, all_sst = mhw_utils.load_noaa_sst(all_sst_files)
    else:
        # Kludge
        import pdb; pdb.set_trace()
        ds = xarray.load_dataset(all_sst_files[0])
        lat_coord, lon_coord = ds.lat, ds.lon

    # Time
    time_dict = build_time_dict(t)

    # Scaling
    scls = np.zeros_like(t).astype(float)
    if scale_file is not None:
        # Use scales
        scale_tbl = pandas.read_hdf(scale_file, 'median_climate')
        for kk, it in enumerate(t):
            mtch = np.where(scale_tbl.index.to_pydatetime() == datetime.datetime.fromordinal(it))[0][0]
            scls[kk] = scale_tbl.medSSTa_savgol[mtch]


    # Start the db's
    if os.path.isfile(climate_db_file):
        os.remove(climate_db_file)

    # Main loop
    if cut_sky:
        irange = np.arange(355, 365)
        jrange = np.arange(715,725)
    else:
        irange = np.arange(lat_coord.shape[0])
        jrange = np.arange(lon_coord.shape[0])
    ii_grid, jj_grid = np.meshgrid(irange, jrange)
    ii_grid = ii_grid.flatten()
    jj_grid = jj_grid.flatten()
    if n_calc is None:
        n_calc = len(irange) * len(jrange)

    # Init
    lenClimYear = 366  # This has to match what is in climate.py
    out_seas = np.zeros((lenClimYear, lat_coord.shape[0], lon_coord.shape[0]), dtype='float32')
    out_thresh = np.zeros((lenClimYear, lat_coord.shape[0], lon_coord.shape[0]), dtype='float32')

    counter = 0
    tot_events = 0

    # Init climate items

    # Length of climatological year
    lenClimYear = 366
    feb29 = 60
    # Window
    windowHalfWidth=5
    wHW_array = np.outer(np.ones(1000, dtype='int'), np.arange(-windowHalfWidth, windowHalfWidth + 1))

    # Inialize arrays
    thresh_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')
    seas_climYear = np.NaN * np.zeros(lenClimYear, dtype='float32')

    doyClim = time_dict['doy']
    TClim = len(doyClim)

    clim_start = 0
    clim_end = len(doyClim)
    nwHW = wHW_array.shape[1]

    # Smoothing
    smoothPercentile = True
    smoothPercentileWidth = 31

    # Main loop
    while (counter < n_calc):
        # Init
        thresh_climYear[:] = np.nan
        seas_climYear[:] = np.nan

        ilat = ii_grid[counter]
        jlon = jj_grid[counter]
        counter += 1

        # Grab SST values
        SST = mhw_utils.grab_T(all_sst, ilat, jlon)
        frac = np.sum(np.invert(SST.mask))/t.size
        if SST.mask is np.bool_(False) or frac > min_frac:
            pass
        else:
            continue

        # Work it
        SST -= scls
        mhw_numba.calc_clim(lenClimYear, feb29, doyClim, clim_start, 
                            clim_end, wHW_array, nwHW, TClim, 
                            thresh_climYear, SST, pctile, 
                            seas_climYear)
        # Leap day
        thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
        seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]


        # Smooth if desired
        if smoothPercentile:
            thresh_climYear = mhw_utils.runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = mhw_utils.runavg(seas_climYear, smoothPercentileWidth)

        # Save
        out_seas[:, ilat, jlon] = seas_climYear
        out_thresh[:, ilat, jlon] = thresh_climYear

        # Cubes
        if (counter % 100000 == 0) or (counter == n_calc):
            print('count={} of {}.'.format(counter, n_calc))
            print("Saving...")
            time_coord = xarray.IndexVariable('doy', np.arange(366, dtype=int) + 1)
            da_seasonal = xarray.DataArray(out_seas, coords=[time_coord, lat_coord, lon_coord])
            da_thresh = xarray.DataArray(out_thresh, coords=[time_coord, lat_coord, lon_coord])
            # Data set
            climate_ds = xarray.Dataset({"seasonalT": da_seasonal,
                                         "threshT": da_thresh})
            # Write
            climate_ds.to_netcdf(climate_db_file)#, encoding=encoding)

            print("Wrote: {}".format(climate_db_file))

    print("All done!!")



def build_time_dict(t):
    """
    Generate a time dict for guiding climate analysis

    Parameters
    ----------
    t : np.ndarray
        Array or ordinal values of time

    Returns
    -------
    times : dict

    """

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

def noaa_median_sst(outfile, climate_file=None, 
                    years=(1983, 2019), check=True):
    """
    Calculate the global ocean's SST evolution across
    the time period.
    
    Filter with a savgol and write to disk

    Parameters
    ----------
    outfile : str
        Output file of the goga
    climate_file : str, optional
        Climatology file to be analyzed
        Default is NOAA_OI_climate_1983-2019.nc
    years : tuple, optional
        Years to analyze
    check : bool, optional
        Debug?

    """
    feb29 = 60  
    # Load
    if climate_file is None:
        climate_file = os.path.join(os.getenv('NOAA_OI'), 
                                    'NOAA_OI_climate_1983-2019.nc')
    ds = xarray.open_dataset(climate_file)
    sT_data = ds.seasonalT.values

    # Run it
    sv_yr, sv_dy, sv_medSST, sv_medSSTa = [], [], [], []
    for year in range(years[0], years[1] + 1):
        print('year={}'.format(year))
        # Load
        noaa_file = os.path.join(os.getenv('NOAA_OI'), 'sst.day.mean.{}.nc'.format(year))
        sst_ds = xarray.open_dataset(noaa_file)
        SST = sst_ds.sst.to_masked_array()
        # Loop on days
        for day in range(SST.shape[0]):
            # print('day={}'.format(day))
            SSTd = SST[day, :, :]
            sv_yr.append(year)
            sv_dy.append(day + 1)  # Jan 1 = 1
            # Stats
            sv_medSST.append(np.median(SSTd[~SSTd.mask]))
            # import pdb; pdb.set_trace()
            # Deal with leap year
            offset = 0
            if ((year - 1984) % 4) != 0 and (day >= feb29):
                offset = 1
            # SSTa
            SSTa = SSTd - sT_data[day + offset, :, :]
            sv_medSSTa.append(np.median(SSTa[~SSTd.mask]))

    # Dates
    tdates = [datetime.datetime(year, 1, 1) + datetime.timedelta(days=day - 1) for year, day in zip(sv_yr, sv_dy)]

    # Pandas
    pd_dict = dict(date=tdates, medSST=sv_medSST, medSSTa=sv_medSSTa)
    pd_tbl = pandas.DataFrame(pd_dict)
    pd_tbl = pd_tbl.set_index('date')

    # Savgol
    SSTa_filt = signal.savgol_filter(sv_medSSTa, 365, 3)
    pd_tbl['medSSTa_savgol'] = SSTa_filt

    # Check?
    if check:
        import matplotlib
        from matplotlib import pyplot as plt
        #
        plt.clf()
        ax = plt.gca()
        #
        dates = matplotlib.dates.date2num(pd_tbl.index)

        ax.plot_date(dates, sv_medSSTa)
        ax.plot_date(dates, SSTa_filt, 'r-')
        # matplotlib.pyplot.plot_date(dates, values)
        #
        ax.set_ylabel('Median SSTa')
        ax.set_xlabel('Year')
        #
        plt.show()
        embed(header='334 of climate')

    # Save pandas
    pd_tbl.to_parquet(outfile)#, 'median_climate', mode='w')
    #pd_tbl.to_hdf(outfile, 'median_climate', mode='w')
    print("Wrote: {}".format(outfile))

# Command line execution
if __name__ == '__main__':

    # Test
    if False:
        noaa_seas_thresh('test.nc',
                         climatologyPeriod=(1983, 1988),
                         cut_sky=False)
    # Traditional Climate
    if False:
        noaa_seas_thresh('/home/xavier/Projects/Oceanography/MHWs/db/NOAA_OI_climate_1983-2012.nc',
                         climatologyPeriod=(1983, 2012),
                         cut_sky=False)

    # Full Climate 1983-2019; not scaled
    if False:
        noaa_seas_thresh('/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_climate_1983-2019.nc',
                         climatologyPeriod=(1983, 2019),
                         cut_sky=False)

    # Median SSTa (savgol)
    if False:
        noaa_median_sst('data/climate/noaa_median_climate_1983_2019.hdf', years=(1983,2019))
        # TEST
        #climate_file = os.path.join(os.getenv('NOAA_OI'), 
        #                            'NOAA_OI_climate_1983-2012.nc')
        #noaa_median_sst('data/climate/test.parquet', 
        #                climate_file=climate_file,
        #                years=(1983,1984))

    # Test scaled
    if False:
        scale_file = os.path.join(resource_filename('mhw', 'data'), 'climate',
                                  'noaa_median_climate_1983_2012.hdf')
        noaa_seas_thresh('test_scaled.nc',
                         climatologyPeriod=(1983, 1985),
                         cut_sky=False, scale_file=scale_file)
    # Full; scaled
    if False:
        scale_file = os.path.join(resource_filename('mhw', 'data'), 'climate',
                                  'noaa_median_climate_1983_2019.hdf')
        noaa_seas_thresh(
            '/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019.nc',
            climatologyPeriod=(1983, 2019),
            cut_sky=False, scale_file=scale_file)

    # 95 percentile
    if False:
        pctile = 95.
        # Full Climate 1983-2019, 95th precentile; not scaled
        noaa_seas_thresh('/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_climate_1983-2019_95.nc',
                         climatologyPeriod=(1983, 2019),
                         cut_sky=False, pctile=pctile)

        # scaled
        scale_file = os.path.join(resource_filename('mhw', 'data'), 'climate',
                                  'noaa_median_climate_1983_2019.hdf')
        noaa_seas_thresh(
            '/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019_95.nc',
            climatologyPeriod=(1983, 2019),
            cut_sky=False, scale_file=scale_file, pctile=pctile)

    # 10 percentile
    if False:
        pctile = 10.

        '''
        # Raw
        noaa_seas_thresh(
            '/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_climate_1983-2019_10.nc',
            climatologyPeriod=(1983, 2019), cut_sky=False, pctile=pctile)
        '''

        # scaled
        scale_file = os.path.join(resource_filename('mhw', 'data'), 'climate',
                                  'noaa_median_climate_1983_2019.hdf')
        noaa_seas_thresh(
            '/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019_10.nc',
            climatologyPeriod=(1983, 2019),
            cut_sky=False, scale_file=scale_file, pctile=pctile)
