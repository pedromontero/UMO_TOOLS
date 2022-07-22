#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_concatenate.py

@Purpose: Concatenate a list of netcdf files along a dim
@version: 1.0

@python version: 3.9
@author: Pedro Montero
@license: INTECMAR
@requires: xarray

@date 2022/07/19

@history:


"""

import xarray as xr
from datetime import datetime, timedelta
from common import read_input





class NcConcatDimension:

    def __init__(self, file_in_list,  dimension, file_out):
        self.file_in_list = file_in_list
        self.dimension = dimension
        self.file_out = file_out
        self.ds_out = None


    def write_ds(self):
        self.ds_out.to_netcdf(self.file_out)
        print(f'NETcdf output: {self.file_out}')

    def get_array(self, name):
        return self.ds.Datarray()

    def get_ds_list(self):
        return [xr.open_dataset(file_in) for file_in in self.file_in_list]

    def concat_dimension(self):
        ds_list = self.get_ds_list()
        self.ds_out = xr.concat(ds_list, self.dimension)
        self.write_ds()


def get_file_in_list(file_in_prefix, initial_date_str, number_days):
    initial_date = datetime.strptime(initial_date_str, '%Y-%m-%d')
    list_of_dates = []
    for number_day in range(0,number_days):
        for hour in range(0,24):
            date_file = initial_date + timedelta(days=number_day)+ timedelta(hours=hour)
            file_name = date_file.strftime(file_in_prefix + '%Y_%m_%d_%H00.nc')
            list_of_dates.append(file_name)
    return list_of_dates




if __name__ == '__main__':
    # input
    input_keys = ['file_in_prefix', 'initial_date', 'number_days', 'dimension', 'file_out']
    inputs = read_input('nc_concatenate.json', input_keys)
    # end input

    dates_list = get_file_in_list(inputs['file_in_prefix'],
                                  inputs['initial_date'],
                                  inputs['number_days'])
    print(dates_list)

    nc = NcConcatDimension(dates_list,
                           inputs['dimension'],
                           inputs['file_out'])
    nc.concat_dimension()
