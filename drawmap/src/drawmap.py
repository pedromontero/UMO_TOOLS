#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
drawmap.py

@Purpose: draw a current map from a hdf mohid or netcdfoutput
@version: 1.0

@python version: 3.9
@author: Pedro Montero
@license: INTECMAR
@requires: matplotlib, numpy, toolkits.basemap

@date 2021/10/13

@history:


"""


from common.readers.reader_factory import read_factory
from common import read_input
from common.boundarybox import BoundaryBox
from drawcurrents import drawcurrents


def read_inputs(input_file):
    """Read keywords for options"""
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
    return read_input(input_file, input_keys)


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
    inputs = read_inputs('drawmap.json')

    file_path = inputs['path_in']
    file_in = inputs['file_in']
    nx = inputs['nx']
    ny = inputs['ny']
    scale = inputs['scale']
    resolution = inputs['resolution']
    level = inputs['n_level']
    time = inputs['n_time']
    file_hdf_out = inputs['file_out']
    file_out = os.path.join(file_path, file_hdf_out)
    title = inputs['title']
    file_name = os.path.join(file_path, file_in)
    style = inputs['style']
    limits = inputs['limits']
    boundary_box = BoundaryBox(limits[0], limits[1], limits[2], limits[3])

    u_name = inputs['u']
    v_name = inputs['v']

    print('Opening: {0}'.format(file_name))

    factory = read_factory(file_name)
    reader = factory.get_reader()

    data = reader.get_date(time)
    data_str = data.strftime("%Y-%m-%d %H:%M UTC")
    data_comp = data.strftime("%Y%m%d%H%M")
    title = title + " " + data_str
    file_out = file_out + '_' + data_comp + '.png'

    lat = reader.latitudes
    lon = reader.longitudes

    u = reader.get_variable(u_name, time)
    v = reader.get_variable(v_name, time)

    if reader.coordinates_rank == 1:
        lats = lat[0:reader.n_latitudes - 1]
        lons = lon[0:reader.n_longitudes - 1]
    elif reader.coordinates_rank == 2:
        lats = lat[0:reader.n_longitudes - 1, 0:reader.n_latitudes - 1]
        lons = lon[0:reader.n_longitudes - 1, 0:reader.n_latitudes - 1]

    if len(u.shape) == 3:
        us = u[level, 0:reader.n_latitudes - 1, 0:reader.n_longitudes - 1]
        vs = v[level, 0:reader.n_latitudes - 1, 0:reader.n_longitudes - 1]
    elif len(u.shape) == 2:
        us = u[0:reader.n_latitudes - 1, 0:reader.n_longitudes - 1]
        vs = v[0:reader.n_latitudes - 1, 0:reader.n_longitudes - 1]

    mod = pow((pow(us, 2) + pow(vs, 2)), .5)

    reader.close()

    drawcurrents(reader.coordinates_rank, nx, ny, scale, resolution, level, time, lats, lons, us, vs, mod, file_out, title, style, boundary_box)


if __name__ == '__main__':
    main()
