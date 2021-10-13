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
import numpy as np
from common import read_input
from common.readers import reader_HDF
from common.readers import reader_NetCDF
from common.boundarybox import BoundaryBox

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
    input_file = 'drawmap_ncdf.json'
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
    boundary_box =BoundaryBox(limits[0], limits[1], limits[2], limits[3])

    u_name = inputs['u']
    v_name = inputs['v']

    print('Opening: {0}'.format(file_name))

    hdf = False
    ncdf = False

    extension = file_hdf.split('.')[1]
    hdf = (extension == 'hdf' or extension == 'hdf5')
    ncdf = (extension == 'nc' or extension == 'nc4')
    print(hdf, ncdf)

    if hdf:
        reader = reader_HDF.ReaderHDF(file_name)
    elif ncdf:
        reader = reader_NetCDF.ReaderNetCDF(file_name)

    data = reader.get_date(time)
    data_str = data.strftime("%Y-%m-%d %H:%M UTC")
    data_comp = data.strftime("%Y%m%d%H%M")
    title = title + " " + data_str
    file_out = file_out + '_' + data_comp + '.png'

    lat = reader.latitudes
    lon = reader.longitudes

    u = reader.get_variable(u_name, time)
    v = reader.get_variable(v_name, time)

    nlon = reader.n_longitudes
    nlat = reader.n_latitudes

    if reader.coordinates_rank == 1:
        lats = lat[0:nlat - 1]
        lons = lon[0:nlon - 1]
    elif reader.coordinates_rank == 2:
        lats = lat[0:nlon-1, 0:nlat - 1]
        lons = lon[0:nlon-1, 0:nlat - 1]

    if len(u.shape) == 3:
        print(f'entro y nlon = {nlon} y nlat = {nlat}')
        us = u[level, 0:nlat - 1, 0:nlon - 1]
        vs = v[level, 0:nlat - 1, 0:nlon - 1]
    elif len(u.shape) == 2:
        us = u[0:nlat - 1, 0:nlon - 1]
        vs = v[0:nlat - 1, 0:nlon - 1]

    mod = pow((pow(us, 2) + pow(vs, 2)), .5)

    reader.close()

    drawcurrents(reader.coordinates_rank, nx, ny, scale, resolution, level, time, lats, lons, us, vs, mod, file_out, title, style, boundary_box)


if __name__ == '__main__':
    main()
