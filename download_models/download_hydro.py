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
import datetime
import urllib.request
from collections import namedtuple


def main():
    """
    Main program.

    Given a date (date_ini) and a number of days, retrieve these hydrodynamic files and wrf files and copy in a target
    folder, ./in
    Range: since date_ini to date_ini+ number of days, last included
    hist_log is a logical var indicating if the thredds folder is historical

    """
    # input
    type_url = 'hydro_hist'
    name_grid = 'AROUSA'
    date_ini = datetime.date(2017, 3, 30)
    days = 0
    # end input

    date_fin = date_ini + datetime.timedelta(days=days)
    dates = [date_ini + datetime.timedelta(days=d) for d in range((date_fin - date_ini).days + 1)]

    os.chdir(r'.\..')
    root = os.getcwd()
    path_in = os.path.join(root, 'datos/download_models')

    for n, date in enumerate(dates):

        url = Url()
        ffile_in = os.path.join(path_in, url.get_file(type_url, name_grid, date))
        print(ffile_in)
        if os.path.exists(ffile_in):
            print('xa atopei o ficheiro', ffile_in)
        else:
            url_grid = url.get_url(type_url, name_grid, date)
            print('Voy baixar {0} a {1}'.format(url_grid, ffile_in))
            urllib.request.urlretrieve(url_grid, ffile_in)

    print('End')


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

    def get_url(self, type_url,name_grid, date):

        grid = self.GRID[name_grid]
        URL = {'hydro_raw':
           f'http://mandeo.meteogalicia.es/thredds/fileServer/modelos/mohid/rawoutput/{grid.grid}/%Y%m%d/',
           'hydro_hist':
           f'http://mandeo.meteogalicia.es/thredds/fileServer/modelos/mohid/history/{grid.grid}/',
           'hydro_nc':
           f'http://mandeo.meteogalicia.es/thredds/fileServer/mohid_{grid.grid_mohid}/files/%Y%m%d/'}
        return date.strftime(URL[type_url])+self.get_file(type_url, name_grid, date)

    def get_file(self, type_url,name_grid, date):

        file = {'hydro_raw':
           f'MOHID_Hydrodynamic_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.hdf5',
           'hydro_hist':
           f'MOHID_Hydrodynamic_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.hdf5',
           'hydro_nc':
           f'MOHID_{self.GRID[name_grid].capital_grid}_%Y%m%d_0000.nc4'}
        return date.strftime(file[type_url])

if __name__ == '__main__':

   main()
