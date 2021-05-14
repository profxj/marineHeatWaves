""" Code to generate complete MHW Event database (SQL) """
import glob
import os
import numpy as np
from pkg_resources import resource_filename
import datetime

import sqlalchemy
from datetime import date
import pandas
import xarray

from mhw import marineHeatWaves
from mhw import utils as mhw_utils

from IPython import embed

MHW_path = os.getenv('MHW')

def build_mhw_events(dbfile, years, noaa_path=None, climate_cube_file=None,
         cut_sky=False, data_in=None, min_frac=0.9, scale_file=None,
         n_calc=None, append=False, seas_climYear=None, 
         thresh_climYear=None, minDT=None,
         detrend_local=False,
         nsub=49999, coldSpells=False, interpolated=False):
    """
    Generate an MHW Event database

    Parameters
    ----------
    dbfile : str
        Output filename for the SQL database
    years : tuple
        Start, end year of database.  Inclusive
    noaa_path : str, optional
        If None, taken from $NOAA_OI
    climate_cube_file : str, optional
        File including the T threshold from climatology
    cut_sky : bool, optional
        Use a subset of the sky for testing
    data_in : tuple, optional
        Loaded SST data.
        lat_coord, lon_coord, t, all_sst
    interpolated : bool, optional
        Interpolated SST files?
    min_frac
    n_calc
    append
    nsub : int, optional
        How many patches to analyze before writing to disk
    seas_climYear
    thresh_climYear
    scale_file : str, optional
        if set, scale the climate by the varying climate as recorded
        in the provided NOAA pandas Table
        Note: One should also be using a varying climate_cube_file
    coldSpells : bool, optional
        If True, search for Cold spells, not Heat Wave events
    minDT : float, optional
        If set, a MHWE must exceed this absoulte DT
    detrend_local : bool, optional
        If True, detrend the climatology at each grid cell with a linear trend

    Returns
    -------

    """
    # Path
    if noaa_path is None:
        if interpolated:
            noaa_path = os.path.join(os.getenv("NOAA_OI"), 'Interpolated')
            sst_root='interpolated_sst*'
        else:
            noaa_path = os.getenv("NOAA_OI")
            sst_root='sst.day.*nc'

    # Load climate
    if climate_cube_file is None:
        climate_cube_file = os.path.join(noaa_path, 'NOAA_OI_climate_1983-2012.nc')
    if seas_climYear is None or thresh_climYear is None:
        print("Loading the climate: {}".format(climate_cube_file))
        ds = xarray.open_dataset(climate_cube_file)
        seas_climYear = ds.seasonalT
        thresh_climYear = ds.threshT
        # No lazy
        _ = seas_climYear.data[:]
        _ = thresh_climYear.data[:]
        # 
        if detrend_local:
            linear_fit = ds.linear
            _ = linear_fit.data[:]

    # Grab the list of SST V2 files
    #all_sst_files = glob.glob(os.path.join(noaa_path, 'sst.day*nc'))
    all_sst_files = glob.glob(os.path.join(noaa_path, sst_root))
    all_sst_files.sort()

    # Load the Cubes into memory
    if data_in is None:
        for ii, ifile in enumerate(all_sst_files):
            if str(years[0]) in ifile:
                istart = ii
            if str(years[1]) in ifile:
                iend = ii
        all_sst_files = all_sst_files[istart:iend+1]

        print("Loading up the files. Be patient...")
        lat_coord, lon_coord, t, all_sst = mhw_utils.load_noaa_sst(
            all_sst_files, interpolated=interpolated)
    else:
        lat_coord, lon_coord, t, all_sst = data_in

    # Day of year
    doy = mhw_utils.calc_doy(t)

    # Scaling
    scls = np.zeros_like(t).astype(float)
    if scale_file is not None:
        # Use scales
        if scale_file[-3:] == 'hdf':
            scale_tbl = pandas.read_hdf(scale_file, 'median_climate')
        elif scale_file[-7:] == 'parquet':
            scale_tbl = pandas.read_parquet(scale_file)
        else:
            raise IOError("Not ready for this type of scale_file: {}".format(scale_file))
        for kk, it in enumerate(t):
            mtch = np.where(scale_tbl.index.to_pydatetime() == datetime.datetime.fromordinal(it))[0][0]
            scls[kk] = scale_tbl.medSSTa_savgol[mtch]

    # Setup for output
    # ints -- all are days
    int_keys = ['time_start', 'time_end', 'time_peak', 'duration', 'duration_moderate', 'duration_strong',
                'duration_severe', 'duration_extreme', 'category', 'n_events']
    float_keys = ['intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative']

    for key in float_keys.copy():
        float_keys += [key+'_relThresh', key+'_abs']
    float_keys += ['rate_onset', 'rate_decline']


    # Init the array
    dtypes = []

    max_events = 1000
    for key in int_keys:
        dtypes += [(key, 'int32', (max_events))]
    for key in float_keys:
        dtypes += [(key, 'float32', (max_events))]
    data = np.empty((nsub,), dtype=dtypes)

    def init_data(idata):
        for key in int_keys:
            idata[key][:][:] = 0
        for key in float_keys:
            idata[key][:][:] = 0.

    init_data(data)

    # Start the db's
    if os.path.isfile(dbfile) and not append:
        # BE CAREFUL!!
        os.remove(dbfile)

    # Connect to the db
    engine = sqlalchemy.create_engine('sqlite:///'+dbfile)
    if append:
        connection = engine.connect()
        metadata = sqlalchemy.MetaData()
        mhw_tbl = sqlalchemy.Table('MHW_Events', metadata, autoload=True, autoload_with=engine)
        query = sqlalchemy.select([mhw_tbl]).where(sqlalchemy.and_(
            mhw_tbl.columns.ievent == 108, mhw_tbl.columns.time_start == 737341, mhw_tbl.columns.duration == 14))
        result = connection.execute(query).fetchall()[-1]
        last_lat, last_lon = result[12:14]
        # Indices
        last_ilat = np.where(lat_coord.data == last_lat)[0][0]
        last_jlon = np.where(lon_coord.data == last_lon)[0][0]

    # Main loop
    if cut_sky:
        irange = np.arange(335, 365)
        jrange = np.arange(715, 755)
    else:
        irange = np.arange(lat_coord.shape[0])
        jrange = np.arange(lon_coord.shape[0])
    ii_grid, jj_grid = np.meshgrid(irange, jrange)
    ii_grid = ii_grid.flatten()
    jj_grid = jj_grid.flatten()
    if n_calc is None:
        n_calc = len(irange) * len(jrange)

    # Last
    if append:
        counter = np.where((ii_grid == last_ilat) & (jj_grid == last_jlon))[0][0]
    else:
        counter = 0

    # Cold spells?
    if coldSpells:
        isign = -1.
    else:
        isign = 1.

    # Main loop
    sub_count = 0
    tot_events = 0
    ilats = []
    jlons = []
    while (counter < n_calc):
        # Load Temperatures
        nmask = 0

        # Slurp
        ilat = ii_grid[counter]
        jlon = jj_grid[counter]
        ilats += [ilat]
        jlons += [jlon]
        counter += 1

        # Ice/land??
        SST = mhw_utils.grab_T(all_sst, ilat, jlon)
        frac = np.sum(np.invert(SST.mask))/t.size
        if SST.mask is np.bool_(False) or frac > min_frac:
            # Scale
            assert SST.size == scls.size # Be wary of masking
            SST -= scls
            if detrend_local:
                f = np.poly1d(linear_fit.data[:, ilat, jlon])
                SST -= f(np.arange(SST.size))
            # Run
            marineHeatWaves.detect_with_input_climate(t, doy,
                                                      isign*SST.flatten(),
                                                      isign*seas_climYear.data[:, ilat, jlon].flatten(),
                                                      isign*thresh_climYear.data[:, ilat, jlon].flatten(),
                                                      sub_count, data,
                                                      minDT=minDT)
        else:
            nmask += 1

        # Save to db
        if (sub_count == nsub-1) or (counter == n_calc):
            print('Count: {}, {}'.format(counter, n_calc))
            # Write
            final_tbl = None
            for kk, iilat, jjlon in zip(range(sub_count), ilats, jlons):
                # Fill me in
                nevent = data['n_events'][kk][0]
                tot_events += nevent
                if nevent > 0:
                    int_dict = {}
                    for key in int_keys:
                        int_dict[key] = data[key][kk][0:nevent]
                    # Ints first
                    sub_tbl = pandas.DataFrame.from_dict(int_dict)
                    # Event number
                    sub_tbl['ievent'] = np.arange(nevent)
                    # Time
                    sub_tbl['date'] = pandas.to_datetime([date.fromordinal(tt) for tt in sub_tbl['time_start']])
                    # Lat, lon
                    sub_tbl['lat'] = [lat_coord.data[iilat]] * nevent
                    sub_tbl['lon'] = [lon_coord.data[jjlon]] * nevent
                    # Floats
                    float_dict = {}
                    for key in float_keys:
                        float_dict[key] = data[key][kk][0:nevent]
                    sub_tbl2 = pandas.DataFrame.from_dict(float_dict)
                    #sub_tbl2 = sub_tbl2.astype('float32')
                    # Final
                    cat = pandas.concat([sub_tbl, sub_tbl2], axis=1, join='inner')
                    if final_tbl is None:
                        final_tbl = cat
                    else:
                        final_tbl = pandas.concat([final_tbl, cat], ignore_index=True)

            # Add to DB
            if final_tbl is not None:
                final_tbl.to_sql('MHW_Events', con=engine, if_exists='append')

            # Reset
            init_data(data)
            sub_count = 0
            ilats = []
            jlons = []
        else:
            sub_count += 1

        # Count
        if (counter % 10000) == 0:
            print('count={} of {}. {} were masked. {} total'.format(
                counter, n_calc, nmask, tot_events))

    print("All done!!")
    print("See {}".format(dbfile))


def main(flg_main):
    if flg_main == 'all':
        flg_main = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_main = int(flg_main)
        
    noaa_path = os.getenv('NOAA_OI')

    # Test
    if flg_main & (2 ** 0):
        # Scaled seasonalT, thresholdT
        '''
        main('tst.db',
             (1983,1985),
             #climate_cube_file='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019.nc',
             scale_file=os.path.join(resource_filename('mhw', 'data'), 'climate',
                                     'noaa_median_climate_1983_2019.hdf'),
             cut_sky=False, append=False)
        '''

        # T10 (Cold waves!)
        build_mhw_events('tst.db', (1983,1985), cut_sky=True, append=False, coldSpells=True,
             nsub=1000,
             climate_cube_file=os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019_10.nc'))

    # Full runs

    # Default run to match Oliver (+ a few extra years)
    if flg_main & (2 ** 1):
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_defaults.db'), 
             (1983,2019), cut_sky=False, append=False)

    # 2019 climate
    if flg_main & (2 ** 2):
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019.db'), 
             (1983,2019), cut_sky=False, append=False,
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019.nc'))

    # 2019, Scaled seasonalT, thresholdT (median)
    if flg_main & (2 ** 3):
        print("Running 2019, de-trended median")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_median.db'),
             (1983,2019),
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_detrend_median_climate_1983-2019.nc'),
             scale_file=os.path.join(noaa_path, 
                                     'noaa_detrend_median_1983_2019.parquet'),
             cut_sky=False, append=False)

    # 2019, Scaled seasonalT, thresholdT (mean)
    if flg_main & (2 ** 4):
        print("Running 2019, de-trended mean")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_mean.db'),
             (1983,2019),
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_detrend_mean_climate_1983-2019.nc'),
             scale_file=os.path.join(noaa_path, 
                                     'noaa_detrend_mean_1983_2019.parquet'),
             cut_sky=False, append=False)

        # T10 (Cold waves!)
        #main('/home/xavier/Projects/Oceanography/MHW/db/mcs_allsky_defaults.db',
        #     (1983,2019), cut_sky=False, append=False, coldSpells=True,
        #     climate_cube_file = os.path.join(os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019_10.nc'))

    # Not smoothed
    if flg_main & (2 ** 5):
        print("Running 2019, not smoothed")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_nosmooth.db'), 
             (1983,2019), cut_sky=False, append=False,
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019_nosmooth.nc'))

    # DT > 0.5K
    if flg_main & (2 ** 6):
        print("Running 2019 with DT>0.5K ")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_DT0.5.db'), 
             (1983,2019), cut_sky=False, append=False,
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_climate_1983-2019.nc'),
             minDT=0.5)

    # de-trend (mean) + DT > 0.5K
    if flg_main & (2 ** 7):
        print("Running 2019 + de-trend mean + DT>0.5K ")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_mean_DT0.5.db'), 
             (1983,2019), cut_sky=False, append=False,
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_detrend_mean_climate_1983-2019.nc'),
             scale_file=os.path.join(noaa_path, 
                                     'noaa_detrend_mean_1983_2019.parquet'),
             minDT=0.5)

    # 2019, detrend linear, local
    if flg_main & (2 ** 8):
        print("Running 2019 with linear de-trend ")
        build_mhw_events(os.path.join(MHW_path, 'db', 
                                      'mhw_events_allsky_2019_local.db'), 
             (1983,2019), 
             cut_sky=False, 
             detrend_local=True,
             climate_cube_file=os.path.join(
                 os.getenv('NOAA_OI'), 'NOAA_OI_detrend_local_climate_1983-2019.nc'))

    # Interpolated (2.5deg) 
    if flg_main & (2 ** 10):
        # Default run to match Oliver (+ a few extra years)
        climate_cube_file = os.path.join(noaa_path, 'Interpolated', 'NOAA_OI_climate_2.5deg_1983-2012.nc')
        build_mhw_events('/home/xavier/Projects/Oceanography/MHW/db/mhw_events_interp2.5_2012.db',
                            (1982,2019), cut_sky=False, append=False,
                            interpolated=True,
                            climate_cube_file=climate_cube_file)

        # 1983-2019 climatology
        climate_cube_file = os.path.join(noaa_path, 'NOAA_OI_climate_2.5deg_1983-2019.nc')
        build_mhw_events('/home/xavier/Projects/Oceanography/MHW/db/mhw_events_interp2.5_2019.db',
                            (1982,2019), cut_sky=False, append=False,
                            interpolated=True,
                            climate_cube_file=climate_cube_file)
                            
    # Parquet files
    if flg_main & (2 ** 50):
        clobber = False
        possible_files = ['mhw_events_allsky_defaults.db', 
                          'mhw_events_allsky_2019.db',
                          'mhw_events_allsky_2019_median.db',
                          'mhw_events_allsky_2019_mean.db',
                          'mhw_events_allsky_2019_nosmooth.db',
                          'mhw_events_allsky_2019_DT0.5.db', 
                          'mhw_events_allsky_2019_mean_DT0.5.db', 
                          'mhw_events_allsky_2019_local.db',
        ]
        for root in possible_files:
            ifile = os.path.join(MHW_path, 'db', root)
            if not os.path.isfile(ifile):
                print("Input file not present: {}".format(ifile))
                print("Skipping")
                continue
            # Output file
            pfile = ifile.replace('.db', '.parquet')
            if os.path.isfile(pfile) and not clobber:
                print("Not clobbering: {}".format(pfile))
                continue
            # Load
            print("Loading the events from: {}".format(ifile))
            engine = sqlalchemy.create_engine('sqlite:///'+ifile)
            mhw_events = pandas.read_sql_table('MHW_Events', con=engine)
            # Write to parquet
            mhw_events.to_parquet(pfile)
            print("Wrote: {}".format(pfile))


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg_main = 0
        #flg_main += 2 ** 1  # Hobday
        #flg_main += 2 ** 2  # 2019
        #flg_main += 2 ** 3  # De-trended 2019, median
        #flg_main += 2 ** 4  # De-trended 2019, mean
        #flg_main += 2 ** 5  # 2019, not smoothed
        #flg_main += 2 ** 6  # 2019, DT > 0.5K
        #flg_main += 2 ** 7  # De-trended 2019 (mean), DT > 0.5K
        #flg_main += 2 ** 8  # De-trended 2019, linear+local
        #flg_main += 2 ** 10  # Interpolated
        flg_main += 2 ** 50  # Parquet em
    else:
        flg_main = sys.argv[1]

    main(flg_main)