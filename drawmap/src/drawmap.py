#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
drawmap.py

@Purpose: draw a current map from a hdf mohid output
@version: 0.1

@python version: 3.4
@author: Pedro Montero
@license: INTECMAR
@requires: intecmar.fichero, h5py, matplotlib, numpy, toolkits.basemap

@date 2015/07/03

@history:


"""
import h5py
import numpy as np
from datetime import datetime

# from customize import add_lib
# add_lib()

# from intecmar.fichero import input_file

from common import read_input
from drawcurrents import drawcurrents


def main():
    """
    Main program:


    :return:
    """
    import os

    # Start
    print("________________________________________\n")
    print("               DRAWMAP")
    print("________________________________________\n")


    # Read input file
    input_file = 'drawmap.json'
    input_keys = ['path_in',
                  'file_in',
                  'file_out',
                  'nx',
                  'ny',
                  'resolution',
                  'scale',
                  'n_time',
                  'n_level',
                  'title',
                  'style',
                  'limits']
    inputs = read_input(input_file, input_keys)

    file_path = inputs['path_in']
    file_hdf = inputs['file_in']
    nx = inputs['nx']
    ny = inputs['ny']
    scale = inputs['scale']
    resolution = inputs['resolution']
    level = inputs['n_level']
    time = inputs['n_time']
    file_hdf_out = inputs['file_out']
    file_out = os.path.join(file_path, file_hdf_out)
    title = inputs['title']
    file_name = os.path.join(file_path, file_hdf)
    style = inputs['style']
    limits = inputs['limits']

    print('Opening: {0}'.format(file_name))

    f = h5py.File(file_name, "r")

    date = f['/Time/Time_0000' + time]
    ano = int(date[0])
    mes = int(date[1])
    dia = int(date[2])
    hora = int(date[3])
    minuto = int(date[4])

    data = datetime(year=ano, month=mes, day=dia, hour=hora, minute=minuto)
    data_str = data.strftime("%Y-%m-%d %H:%M UTC")
    print(data_str)
    data_comp = data.strftime("%Y%m%d%H%M")
    title = title + " " + data_str
    file_out = file_out + '_' + data_comp + '.png'

    lat = f['/Grid/Latitude']
    lon = f['/Grid/Longitude']

    u = f['/Results/velocity U/velocity U_0000' + time]
    v = f['/Results/velocity V/velocity V_0000' + time]

    nlon = lat.shape[0]
    nlat = lon.shape[1]

    lats = lat[0, 0:nlat - 1]
    lons = lon[0:nlon - 1, 0]

    us = u[level, 0:nlon - 1, 0:nlat - 1]
    vs = v[level, 0:nlon - 1, 0:nlat - 1]
    ust = np.transpose(us)
    vst = np.transpose(vs)

    mod = pow((pow(ust, 2) + pow(vst, 2)), .5)

    f.close()

    drawcurrents(nx, ny, scale, resolution, level, time, lats, lons, ust, vst, mod, file_out, title, style, limits)


if __name__ == '__main__':
    main()
