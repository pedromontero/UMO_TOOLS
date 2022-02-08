#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
radarhf_ui.py

@Purpose: calculate the upwelling index from a total radarhf netcdf file and write it in another netcdf file
@version: 1.0

@python version: 3.9
@author: Pedro Montero
@license: INTECMAR
@requires: json, datetime, pandas, numpy

@date 2022/02/08

@history:

"""

import os
import json
import math
from datetime import datetime, timedelta, timezone
import pandas as pd
import xarray as xr
import numpy as np
from common import read_input


def read_inputs(input_file):
    """Read keywords for options"""
    input_keys = ['path_in',
                  'file_in',
                  'path_out'
                  'file_out'
                  ]
    return read_input(input_file, input_keys)


def radarhf_ui(file_in, file_out):
    """
    main program: read file_in and create the upwelling index and write it in file_out

    :parameter file_in: str with the full path and netcdf file name of the HF radar total file
    :parameter file_out: str with the full path and netcdf file name where to write the upwelling index
    :return:
    """
    ui = UI(file_in)
    ui.create_ui(file_out)

  
class UI:
    """Class for upwelling index field"""

    def __init__(self, file_in):
        self.dataset_in = self.read(file_in)
        self.attributes_dictionary = self.read_attributes('ui_attributes.json')
        self.global_attributes_dictionary = self.read_attributes('ui_global_attributes.json')
        self.dataset_out = None
        self.array = None
        self.time_stamp = (pd.Timestamp(self.dataset_in['TIME'].values[0]))

    @staticmethod
    def read_attributes(attributes_file):
        """Get attributes from a json file
        :parameter attributes_file: json file of attributes

        :return attributes: dictionary with ui attributes"""
        f = open(attributes_file)
        attributes = json.loads(f.read())
        f.close()
        return attributes

    @staticmethod
    def read(file_name):
        ds = xr.open_dataset(file_name)
        return ds

    def write_attributes(self):
        for key, values in self.attributes_dictionary.items():
            self.array.attrs[key] = values

    def write_global_attributes(self):
        for key, values in self.global_attributes_dictionary.items():
            self.dataset_out.attrs[key] = values

        self.dataset_out.attrs['id'] = 'HFR-Galicia-IU_%sZ' % self.time_stamp.isoformat()

        self.dataset_out.attrs['time_coverage_start'] = '%sZ' % (self.time_stamp - timedelta(minutes=30)).isoformat()
        self.dataset_out.attrs['time_coverage_end'] = '%sZ' % (self.time_stamp + timedelta(minutes=30)).isoformat()

        now_utc = datetime(*datetime.now(timezone.utc).timetuple()[0:6]).isoformat()
        self.dataset_out.attrs['date_created'] = '%sZ' % now_utc
        self.dataset_out.attrs['metadata_date_stamp'] = '%sZ' % now_utc
        self.dataset_out.attrs['date_modified'] = '%sZ' % now_utc
        self.dataset_out.attrs['date_issued'] = '%sZ' % now_utc
        self.dataset_out.attrs['history'] = f'{self.time_stamp.isoformat()}Z data collected. {now_utc}Z netCDF file created'

    def calculate_ui(self):
        u = self.dataset_in['EWCT'].values
        v = self.dataset_in['NSCT'].values
        latitudes = self.dataset_in['LATITUDE'].values
        ro = 1025
        cd = 0.0014
        roa = 1.22
        omega = 0.000072921
        # sin_lat42 = 0.669131
        sin_lat = np.sin(math.pi*latitudes/180.)
        f = 2 * omega * sin_lat
        f_stack = np.transpose(f*np.ones((47, 81)))
        upwelling_index = -(roa * cd * (v * ((u * u + v * v) ** 0.5) * 3000000)) / (f_stack * ro)
        return upwelling_index

    def create_ui(self, file_out):
        upwelling_index = self.calculate_ui()
        dtype_ui = "float32"
        conversor = np.dtype(dtype_ui)
        upwelling_index = upwelling_index.astype(conversor)

        self.array = xr.DataArray(upwelling_index, dims=['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'])
        self.write_attributes()
        self.dataset_out = xr.Dataset({'TIME': self.dataset_in['TIME'],
                                  'DEPH': self.dataset_in['DEPH'],
                                  'LATITUDE': self.dataset_in['LATITUDE'],
                                  'LONGITUDE': self.dataset_in['LONGITUDE'],
                                  'UI': self.array})
        self.write_global_attributes()
        self.dataset_out.reset_coords(drop=False).to_netcdf(file_out)


if __name__ == '__main__':
    # Read input file
    inputs = read_inputs('radarhf_ui.json')
    nc_file_in = os.path.join(inputs['path_in'], inputs['file_in'])
    nc_file_out = os.path.join(inputs['path_out'], inputs['file_out'])
    radarhf_ui(nc_file_in, nc_file_out)
