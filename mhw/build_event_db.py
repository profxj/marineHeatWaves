""" Code to generate complete MHW Event database (SQL) """
import glob
import os
import numpy as np
from pkg_resources import resource_filename
import datetime

import sqlalchemy
from datetime import date
import pandas

from mhw import marineHeatWaves
from mhw import utils as mhw_utils
import iris

from IPython import embed

def main(dbfile, years, noaa_path=None, climate_cube_file=None,
             cut_sky=False, all_sst=None, min_frac=0.9, scale_file=None,
             n_calc=None, append=False, seas_climYear=None, thresh_climYear=None):
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
    climate_cube_file
    cut_sky : bool, optional
    all_sst
    min_frac
    n_calc
    append
    seas_climYear
    thresh_climYear
    scale_file : str, optional
        if set, scale the climate by the varying climate as recorded
        in the provided NOAA pandas Table
        One should also be using a varying climate_cube_file

    Returns
    -------

    """
    # Path
    if noaa_path is None:
        noaa_path = os.getenv("NOAA_OI")

    # Load climate
    if climate_cube_file is None:
        climate_cube_file = os.path.join(noaa_path, 'NOAA_OI_climate_1983-2012.nc')
    if seas_climYear is None or thresh_climYear is None:
        print("Loading the climate: {}".format(climate_cube_file))
        seas_climYear = iris.load(climate_cube_file, 'seasonalT')[0]
        thresh_climYear = iris.load(climate_cube_file, 'threshT')[0]
        # No lazy
        _ = seas_climYear.data[:]
        _ = thresh_climYear.data[:]

    # Grab the list of SST V2 files
    all_sst_files = glob.glob(os.path.join(noaa_path, 'sst*nc'))
    all_sst_files.sort()
    # Cut on years
    if '1981' not in all_sst_files[0]:
        raise ValueError("Years not in sync!!")

    # Load the Cubes into memory
    if all_sst is None:
        istart = years[0] - 1981
        iend = years[1] - 1981 + 1
        all_sst_files = all_sst_files[istart:iend]

        print("Loading up the files. Be patient...")
        all_sst = mhw_utils.load_noaa_sst(all_sst_files)

    # Coords
    lat_coord = all_sst[0].coord('latitude')
    lon_coord = all_sst[0].coord('longitude')
    #events_coord = iris.coords.DimCoord(np.arange(100), var_name='events')

    # Time
    t = mhw_utils.grab_t(all_sst)
    doy = mhw_utils.calc_doy(t)

    # Scaling
    scls = np.zeros_like(t).astype(float)
    if scale_file is not None:
        # Use scales
        scale_tbl = pandas.read_hdf(scale_file, 'median_climate')
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
    data = np.empty((1000,), dtype=dtypes)

    def init_data(idata):
        for key in int_keys:
            idata[key][:][:] = 0
        for key in float_keys:
            idata[key][:][:] = 0.

    init_data(data)

    # Start the db's
    if os.path.isfile(dbfile) and not append:
        # BE CAREFUL!!
        print("REMOVE THAT DBFILE!!")
        import pdb; pdb.set_trace()  # No longer an option
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
        last_ilat = np.where(lat_coord.points == last_lat)[0][0]
        last_jlon = np.where(lon_coord.points == last_lon)[0][0]

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
            # Run
            marineHeatWaves.detect_with_input_climate(t, doy, SST.flatten(),
                                                   seas_climYear.data[:, ilat, jlon].flatten(),
                                                   thresh_climYear.data[:, ilat, jlon].flatten(),
                                                   sub_count, data)
        else:
            nmask += 1

        # Save to db
        if (sub_count == 999) or (counter == n_calc):
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
                    sub_tbl['lat'] = [lat_coord[iilat].points[0]] * nevent
                    sub_tbl['lon'] = [lon_coord[jjlon].points[0]] * nevent
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
        print('count={} of {}. {} were masked. {} total'.format(
            counter, n_calc, nmask, tot_events))

    print("All done!!")


# Command line execution
if __name__ == '__main__':

    # Test
    if False:
        # Scaled seasonalT, thresholdT
        main('tst.db',
             (1983,1985),
             #climate_cube_file='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019.nc',
             scale_file=os.path.join(resource_filename('mhw', 'data'), 'climate',
                                     'noaa_median_climate_1983_2019.hdf'),
             cut_sky=False, append=False)

    # Full runs
    if True:
        # Default run to match Oliver (+ a few extra years)
        #main('/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db',
        #                    (1982,2019), cut_sky=False, append=False)

        # Scaled seasonalT, thresholdT
        main('/home/xavier/Projects/Oceanography/MHW/db/mhws_allsky_defaults.db',
             (1983,2019),
             climate_cube_file='/home/xavier/Projects/Oceanography/data/SST/NOAA-OI-SST-V2/NOAA_OI_varyclimate_1983-2019.nc',
             scale_file=os.path.join(resource_filename('mhw', 'data'), 'climate',
                                     'noaa_median_climate_1983_2019.hdf'),
             cut_sky=False, append=False)

