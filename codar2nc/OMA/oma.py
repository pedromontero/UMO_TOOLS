# -*- coding: utf-8 -*-

"""Python with OMA structure"""

import os
import numpy as np

import xarray as xr
from collections import OrderedDict

from datetime import datetime, timedelta


class OMA:
    """Class to abstract the OMA file structure"""

    def __init__(self, date_in, grid):
        self.variables = OrderedDict()
        #  Coordinate variables come from a Dataset

        self.variables['TIME'] = grid.TIME
        self.variables['TIME'].data[0] = date_in  # Change the date
        self.variables['DEPH'] = grid.DEPH
        self.variables['LATITUDE'] = grid.LATITUDE
        self.variables['LONGITUDE'] = grid.LONGITUDE

    def to_netcdf(self, path, file):

        self.variables['crs'] = xr.DataArray(np.int16(0), )
        dataset = xr.Dataset(self.variables)

        full_file_out = os.path.join(path, file)
        dataset.reset_coords(drop=False).to_netcdf(full_file_out)


if __name__ == '__main__':

    malla = xr.open_dataset('./datos/HFR-Galicia-Total_2022_04_01_0000.nc')
    data = datetime(2022, 6, 27, 13, 1, 23)
    path_out = './datos'
    file_out = 'test.nc'

    oma = OMA(data, malla)
    oma.to_netcdf(path_out, file_out)





