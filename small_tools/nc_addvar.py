#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_dropdimension.py

@Purpose: add a var simmilar to other one and fill it with zeros
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


class NcAddVar:

    def __init__(self, file_in, new_var, var_alike, file_out):
        self.file_in = file_in
        self.new_var = new_var
        self.var_alike = var_alike
        self.file_out = file_out
        self.ds = None

    def get_ds(self):
        return xr.open_dataset(self.file_in)

    def write_ds(self):
        self.ds_new.to_netcdf(self.file_out)
        print(f'NETcdf output: {self.file_out}')

    def add_var(self):
        self.ds_new = self.ds.assign(DEPTH=self.ds[self.var_alike] * 0)

    def get_array(self, name):
        return self.ds.Datarray()

    def add_variable(self):
        self.ds = self.get_ds()
       # self.ds.squeeze('DEPTH')
        print(self.ds)
        self.add_var()

        print('\n\n')

        print(self.ds_new)
        self.write_ds()


if __name__ == '__main__':
    # input
    input_keys = ['file_in', 'new_var', 'var_alike', 'file_out']
    inputs = read_input('nc_addvar.json', input_keys)
    # end input

    nc = NcAddVar(inputs['file_in'],
                  inputs['new_var'],
                  inputs['var_alike'],
                  inputs['file_out'])

    nc.add_variable()
