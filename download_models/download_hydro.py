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

class PathIn:

    file_type = {'hydro', 'hydro_hist'}
    prefix: str
    sufix: str



def main():
    """
    Main program.

    Given a date (date_ini) and a number of days, retrieve these hydrodynamic files and wrf files and copy in a target
    folder, ./in
    Range: since date_ini to date_ini+ number of days, last included
    hist_log is a logical var indicating if the thredds folder is historical

    """
    # input
    hydro_log = True

    hist_hydro_log = True
    hist_met_log = False
    date_ini = datetime.date(2020, 2, 20)
    days = 1
    # end input

    date_fin = date_ini + datetime.timedelta(days=days)
    dates = [date_ini + datetime.timedelta(days=d) for d in range((date_fin - date_ini).days + 1)]


    prefix_hydro = 'MOHID_Hydrodynamic_Arousa_'
    sufix_hydro = '_0000.hdf5'

    if hist_hydro_log:
        prefix_url_hydro = r'http://mandeo.meteogalicia.es/thredds/fileServer/modelos/mohid/history/arousa/'
    else:
        prefix_url_hydro = r'http://mandeo.meteogalicia.es/thredds/fileServer/mohid_arousa/fmrc/files/'


    os.chdir(r'.\..')
    root = os.getcwd()
    path_in = os.path.join(root, 'datos/download_models')

    # Download hydro
    if hydro_log:
        for n, date in enumerate(dates):
            file_in = prefix_hydro + date.strftime('%Y%m%d') + sufix_hydro
            ffile_in = os.path.join(path_in, file_in)
            print(ffile_in)
            if os.path.exists(ffile_in):
                print('xa atopei o ficheiro', ffile_in)
            else:
                if hist_hydro_log:
                    url = prefix_url_hydro + file_in
                else:
                    url = prefix_url_hydro + date.strftime('%Y%m%d') + '/' + file_in
                print('Voy baixar {0} a {1}'.format(url, ffile_in))
                urllib.request.urlretrieve(url, ffile_in)

    print('End')


if __name__ == '__main__':
    main()
