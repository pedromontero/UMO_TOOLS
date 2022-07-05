# -*- coding: utf-8 -*-

"""Python with OMA structure"""

import os
import numpy as np

import xarray as xr
from collections import OrderedDict

from datetime import datetime
import json


class OMA:
    """Class to abstract the OMA file structure"""

    def __init__(self, date_in, grid):

        self.variables = OrderedDict()

        #  Coordinate variables come from a Dataset
        self.variables['TIME'] = grid.TIME #.copy(deep=True, data=np.array([date_in]))
        self.variables['TIME'].data[0] = date_in  # Change the date
        print(self.variables['TIME'].values)
        self.variables['DEPH'] = grid.DEPH
        self.variables['LATITUDE'] = grid.LATITUDE
        self.variables['LONGITUDE'] = grid.LONGITUDE
        self.variables['crs'] = grid.crs
        #  self.variables['crs'] = xr.DataArray(np.int16(0), )

        self.variables['TIME_QC'] = grid.TIME_QC
        self.variables['DEPH_QC'] = grid.DEPH_QC

        self.variables['POSITION_QC'] = grid.POSITION_QC
       # self.variables['POSITION_QC'].reindex(TIME=self.variables['TIME'].values)
        del self.variables['POSITION_QC'].attrs['coordinates']
        self.variables['POSITION_QC'].attrs['coordinates'] = 'LATITUDE LONGITUDE DEPH'

        self.variables['EWCT'] = grid.EWCT.copy() #esto no funciona, hay verlo
        #self.variables['EWCT'] = grid.EWCT
        # print('----------------------', self.variables['EWCT'].get_index('TIME'))
        # print(self.variables['TIME'].data)
        self.variables['NSCT'] = grid.NSCT.copy()
        #self.variables['EWCT'].set_index(TIME = {'2022-04-01':'2022-06-27T13:13:00.000000000' })
        #self.variables['EWCT'].set_index(TIME={'TIME': self.variables['TIME'].data})
        # self.variables['NSCT'].reindex(TIME=self.variables['TIME'].values)
        # print(self.variables['TIME'].values)
        #print('----------------------', self.variables['EWCT'].get_index('TIME'))

    def change_data(self, variable, data_in):
        self.variables[variable].data = data_in

    def to_netcdf(self, path, file):

        dataset = xr.Dataset(self.variables)
        full_file_out = os.path.join(path, file)
        dataset.reset_coords(drop=False).to_netcdf(full_file_out)


if __name__ == '__main__':

    malla = xr.open_dataset('./datos/HFR-Galicia-Total_2022_04_01_0000.nc')
    data = datetime(2022, 6, 27, 13, 13)
    path_out = './datos'
    file_out = 'test.nc'

    oma = OMA(data, malla)

    oma.to_netcdf(path_out, file_out)





