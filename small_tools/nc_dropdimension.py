#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_dropdimension.py

@Purpose: remove a dimension from a netcdf file
@version: 1.0

@python version: 3.9
@author: Pedro Montero
@license: INTECMAR
@requires: xarray

@date 2022/07/19

@history:


"""

import xarray as xr
from common import read_input


class NcDropDimension:

    def __init__(self, file_in, dimension, file_out):
        self.file_in = file_in
        self.dimension = dimension
        self.file_out = file_out
        self.ds = None

    def get_ds(self):
        return xr.open_dataset(self.file_in)

    def write_ds(self):
        self.ds.to_netcdf(self.file_out)
        print(f'NETcdf output: {self.file_out}')

    def drop_dim(self):
        del self.ds['DEPH']
         #self.ds.reset_coords('DEPH', drop=True)
        #self.ds.drop_dims(self.dimension)

    def get_array(self, name):
        return self.ds.Datarray()

    def drop_dimension(self):
        self.ds = self.get_ds()
       # self.ds.squeeze('DEPTH')
        print(self.ds)
        self.drop_dim()

        print('\n\n')
        print(self.ds.to_dataframe())
        print(self.ds)
        self.write_ds()



if __name__ == '__main__':
    # input
    input_keys = ['file_in', 'dimension', 'file_out']
    inputs = read_input('nc_dropdimension.json', input_keys)
    # end input

    nc = NcDropDimension(inputs['file_in'],
                                        inputs['dimension'],
                                        inputs['file_out'])
    nc.drop_dimension()