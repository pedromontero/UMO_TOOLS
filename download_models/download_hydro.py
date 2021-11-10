#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
download_hydro.py

@Purpose: Download hydrodynamic Mohid

@python version: 3.7
@author: Pedro Montero
@license: INTECMAR(c)
@requires: os, datetime, urllib

@date: 2018/03/05
@version: 0.7

History:
yyyymmdd INICIAIS MUDANZA

"""

import os
from datetime import datetime, timedelta
import urllib.request
from collections import namedtuple

from common import read_input


def main(input_args) -> None:
    """
    Main program.

    Given a date (date_ini) and a number of days, retrieve these hydrodynamic files and wrf files and copy in a target
    folder, ./in
    Range: since date_ini to date_ini+ number of days, last included
    hist_log is a logical var indicating if the thredds folder is historical

    """

    download_app = DownloadModels(input_args)
    download_app.download_by_dates()

    print('End')


class DownloadModels:

    def __init__(self, input_var):

        self.type_url = input_var['type_url']
        self.name_grid = input_var['name_grid']
        self.date_ini = input_var['date_ini']
        self.days = input_var['days']
        self.path_out = input_var['path_out']
        self.date_ini = self.get_data_ini()
        self.path_out = self.get_path_out()

    def get_data_ini(self):
        return datetime.strptime(self.date_ini, '%Y-%m-%d') + timedelta(-1)\
            if self.type_url == 'hydro' or self.type_url == 'hydro_hist' else 0

    def download_by_dates(self) -> None:
        """ """
        for n, date in enumerate(self.get_dates()):
            url = Url()
            full_file_out = os.path.join(self.path_out, url.get_file(self.type_url, self.name_grid, date))
            if os.path.exists(full_file_out):
                print('xa atopei o ficheiro', full_file_out)
            else:
                url_grid = url.get_url(self.type_url, self.name_grid, date)
                print('Voy baixar {0} a {1}'.format(url_grid, full_file_out))
                urllib.request.urlretrieve(url_grid, full_file_out)

    def get_path_out(self) -> str:
        os.chdir(r'.\..')
        root = os.getcwd()
        return os.path.join(root, self.path_out)

    def get_dates(self) -> list:
        """create a list of dates since date_ini and days"""
        date_fin = self.date_ini + timedelta(days=self.days)
        dates = [self.date_ini + timedelta(days=d) for d in range((date_fin - self.date_ini).days + 1)]
        return dates


class Url:

    """Class for building URL for models"""

    def __init__(self):
        Grid = namedtuple('Grid', ['grid', 'capital_grid', 'grid_mohid'])
        self.GRID = {'ARTABRO': Grid('artabro', 'Artabro', 'artabro'),
                     'AROUSA': Grid('arousa', 'Arousa', 'arousa'),
                     'NOIA': Grid('noia_muros', 'NoiaMuros', 'noiamuros'),
                     'PORTOCORUNA': Grid('porto_coruna', 'CorunaP', 'portocoruna'),
                     'PORTOLANGOSTEIRA': Grid('porto_Langosteira', 'LangosteiraP', 'portolangosteira'),
                     'VIGO': Grid('vigo_pontevedra', 'Vigo', 'vigo')}

    def get_url(self, type_url, name_grid, date):

        grid = self.GRID[name_grid]
        url = {'hydro_raw':
               f'http://mandeo.meteogalicia.es/thredds/fileServer/modelos/mohid/rawoutput/{grid.grid}/%Y%m%d/',
               'hydro_hist':
               f'http://mandeo.meteogalicia.es/thredds/fileServer/modelos/mohid/history/{grid.grid}/',
               'hydro_nc':
               f'http://mandeo.meteogalicia.es/thredds/fileServer/mohid_{grid.grid_mohid}/files/%Y%m%d/'}
        return date.strftime(url[type_url])+self.get_file(type_url, name_grid, date)

    def get_file(self, type_url, name_grid, date):

        file = {'hydro_raw':
           f'MOHID_Hydrodynamic_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.hdf5',
           'hydro_hist':
           f'MOHID_Hydrodynamic_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.hdf5',
           'hydro_nc':
           f'MOHID_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.nc4'}
        return date.strftime(file[type_url])


if __name__ == '__main__':
    # input
    input_keys = ['type_url', 'name_grid', 'date_ini', 'days', 'path_out']
    inputs = read_input('download_hydro.json', input_keys)
    # end input
    main(inputs)
