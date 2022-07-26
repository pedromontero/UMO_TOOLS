#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_changenamevar.py

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


class NcChangeNameVar:

    def __init__(self, file_in, old_name, new_name, file_out):
        self.file_in = file_in
        self.old_name = old_name
        self.new_name = new_name
        self.file_out = file_out
        self.ds = None

    def get_ds(self):
        return xr.open_dataset(self.file_in)

    def write_ds(self):
        self.ds.to_netcdf(self.file_out)
        print(f'NETcdf output: {self.file_out}')

    def change_dim(self):
        rename_dict = {self.old_name: self.new_name}
        print(rename_dict)
        self.ds = self.ds.rename_dims(rename_dict)
        self.ds = self.ds.rename_vars(rename_dict)

    def get_array(self, name):
        return self.ds.Datarray()

    def change_name(self):
        self.ds = self.get_ds()

        print(self.ds)
        self.change_dim()

        print('\n\n *** AFTER CHANGE: ***\n')
        print(self.ds)
        self.write_ds()


if __name__ == '__main__':
    # input
    input_keys = ['file_in', 'old_name','new_name', 'file_out']
    inputs = read_input('nc_changenamevar.json', input_keys)
    # end input

    nc = NcChangeNameVar(inputs['file_in'],
                         inputs['old_name'],
                         inputs['new_name'],
                         inputs['file_out'])
    nc.change_name()
