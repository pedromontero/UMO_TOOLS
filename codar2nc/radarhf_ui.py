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
from datetime import datetime, timedelta, timezone, date, time
import pandas as pd
import xarray as xr
import numpy as np
# from common import read_input

from codar2nc.create_dir_structure import FolderTree


def ui2thredds(number_days=5, end_date=None):

    root_folder = r'../../datos/radarhf/'
    totals_folder = r'dev/RadarOnRAIA/Totals/v2.2'
    ui_folder = r'dev/RadarOnRAIA/UI'
    thredds_folder = os.path.join(root_folder, totals_folder)
    thredds = FolderTree(thredds_folder)

    date_list = get_list_dates(end_date, number_days)

    for date in date_list:
        file_nc = os.path.join(thredds.root, thredds.get_full_total_file_nc(date))
        if check_nc_file(file_nc):

            file_out = date.strftime('HFR-Galicia-UI_%Y_%m_%d_%H00.nc')
            nc_file_out = os.path.join(root_folder, ui_folder, file_out)
            print(date, nc_file_out)
            radarhf_ui(file_nc, nc_file_out)


def ui24_to_thredds(number_days=2, end_date=None):

    root_folder = r'../../datos/radarhf/'
    totals_folder = r'dev/RadarOnRAIA/Totals/v2.2'
    ui_folder = r'dev/RadarOnRAIA/UI'

    date_list = get_list_dates_by_day(end_date, number_days)

    for date in date_list:
        file_out = date.strftime('HFR-Galicia-UI_%Y_%m_%d.nc')
        nc_file_out = os.path.join(root_folder, ui_folder, file_out)
        try:
            radarhf_ui24(date, root_folder, totals_folder, nc_file_out)
        except:
            print('non fixen ', file_out)


def check_nc_file(full_file):
    if os.path.isfile(full_file):
        #print(f'existe {full_file} ')
        return True
    else:
        #print(f'No existe {full_file}')
        return False


def get_list_dates(end_date, number_days):
    if end_date is None:
        end_date = date.today()
    #end_date in this case is tomorrow
    end_date = datetime.combine(end_date + timedelta(1), time(0))
    date_list = [end_date - timedelta(hours=hours) for hours in range(number_days * 24)]

    return date_list


def get_list_dates_by_day(end_date, number_days):

    if end_date is None:
        end_date = date.today()
    # end_date in this case is yesterday
    end_date = datetime.combine(end_date + timedelta(-1), time(0))
    date_list = [end_date - timedelta(days=days) for days in range(number_days)]

    return date_list


def radarhf_ui(file_in, file_out):
    """
    main program: read file_in and create the upwelling index and write it in file_out

    :parameter file_in: str with the full path and netcdf file name of the HF radar total file
    :parameter file_out: str with the full path and netcdf file name where to write the upwelling index
    :return:
    """
    ui = UI(file_in)
    ui.create_ui()
    ui.create_ui_dataset()
    ui.write_ui(file_out)


def radarhf_ui24(day_in, root_folder, totals_folder, file_out):

    mascara = xr.open_dataset(r'./maskout.nc')
    ui24 = UI24(day_in)
    ui24.create_thredds(root_folder, totals_folder)
    ui_dataset = ui24.get_concat_ui()

    mean_ui_dataset = ui_dataset.mean(dim='TIME', keep_attrs=True)
    mean_ui_dataset_min_18h = mean_ui_dataset.where(ui_dataset.count(dim='TIME') > 18)
    mean_ui_dataset_min_18h = mean_ui_dataset_min_18h.where(mascara['MASK'] > 0)

    file_nc = os.path.join(ui24.thredds.root, ui24.thredds.get_full_total_file_nc(day_in))
    if check_nc_file(file_nc):
        ui = UI(file_nc)
        ui.set_ui(np.array([mean_ui_dataset_min_18h['UI']]))
        ui.create_ui_dataset()
        ui.write_ui(file_out)


class UI24:
    """Class for upwelling index from an average of 24 hours"""
    def __init__(self, day):
        self.day = day
        self.thredds = None

    def get_range_of_datetimes(self):
        return [self.day + timedelta(hours=hour) for hour in range(0, 25)]

    def create_thredds(self, root_folder, totals_folder):
        thredds_folder = os.path.join(root_folder, totals_folder)
        self.thredds = FolderTree(thredds_folder)

    def get_concat_ui(self):
        dates_list = self.get_range_of_datetimes()
        ui_all = []
        for date in dates_list:
            file_nc = os.path.join(self.thredds.root, self.thredds.get_full_total_file_nc(date))
            if check_nc_file(file_nc):
                ui = UI(file_nc)
                ui.create_ui()
                ui_all.append(ui.create_ui_dataset())
        return xr.concat(ui_all, dim='TIME')


class UI:
    """Class for upwelling index field"""

    def __init__(self, file_in):
        self.dataset_in = self.read(file_in)
        self.attributes_dictionary = self.read_attributes('ui_attributes.json')
        self.global_attributes_dictionary = self.read_attributes('ui_global_attributes.json')
        self.dataset_out = None
        self.array = None
        self.dtype = 'float32'
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
        for key, value in self.attributes_dictionary.items():
            if isinstance(value, float):
                self.array.attrs[key] = np.float32(value)
            else:
                self.array.attrs[key] = value

        self.array.encoding["scale_factor"] = np.float32(1)
        self.array.encoding["add_offset"] = np.float32(0)
        self.array.encoding["dtype"] = self.dtype
        self.array.encoding["_FillValue"] = "NaN"

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
        self.dataset_out.attrs['history'] = f'{self.time_stamp.isoformat()}Z data collected.' \
                                            f' {now_utc}Z netCDF file created'

    def calculate_ui(self):
        """ Calculate UI for a dataset_in, one time"""

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
        f_stack = np.transpose(f*np.ones((u.shape[3], u.shape[2])))
        upwelling_index = -(roa * cd * (v * ((u * u + v * v) ** 0.5) * 3000000)) / (f_stack * ro)
        return upwelling_index

    def create_ui(self):
        self.upwelling_index = self.calculate_ui()

    def set_ui(self, ui):
        self.upwelling_index = ui

    def create_ui_dataset(self):
        conversor = np.dtype(self.dtype)
        upwelling_index = self.upwelling_index.astype(conversor)
        self.array = xr.DataArray(upwelling_index, dims=['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'])
        self.write_attributes()

        self.dataset_out = xr.Dataset({'TIME': self.dataset_in['TIME'],
                                       'DEPH': self.dataset_in['DEPH'],
                                       'LATITUDE': self.dataset_in['LATITUDE'],
                                       'LONGITUDE': self.dataset_in['LONGITUDE'],
                                       'crs': self.dataset_in['crs'],
                                       'UI': self.array})
        self.write_global_attributes()

        del self.dataset_out['TIME'].attrs['ancillary_variables']
        del self.dataset_out['DEPH'].attrs['ancillary_variables']
        del self.dataset_out['LATITUDE'].attrs['ancillary_variables']
        del self.dataset_out['LONGITUDE'].attrs['ancillary_variables']
        return self.dataset_out

    def write_ui(self, file_out):
        self.dataset_out.reset_coords(drop=False).to_netcdf(file_out)


#def read_inputs(input_file):
#    """Read keywords for options"""
#    input_keys = ['path_in',
#                  'file_in',
#                  'path_out'
#                  'file_out'
#                  ]
#    return read_input(input_file, input_keys)


if __name__ == '__main__':

   # day_in = datetime(2022,4,3)
   # root_folder = r'../../datos/radarhf/'
   # totals_folder = r'dev/RadarOnRAIA/Totals/v2.2'
   # ui_folder = r'dev/RadarOnRAIA/UI'

   # file_out = day_in.strftime('HFR-Galicia-UI_%Y_%m_%d.nc')
   # nc_file_out = os.path.join(root_folder, ui_folder, file_out)


  #  radarhf_ui24(day_in, root_folder, totals_folder, nc_file_out)
    ui24_to_thredds()



    # ui2thredds()

    # Read input file
    # inputs = read_inputs('radarhf_ui.json')
    # nc_file_in = os.path.join(inputs['path_in'], inputs['file_in'])
    # nc_file_out = os.path.join(inputs['path_out'], inputs['file_out'])
    # radarhf_ui(nc_file_in, nc_file_out)
