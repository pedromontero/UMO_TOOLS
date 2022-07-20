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

import os

from common.readers.reader_factory import read_factory
from common import read_input
from common.boundarybox import BoundaryBox
from drawcurrents import drawcurrents



def read_inputs(input_file):
    """Read keywords for options"""
    input_keys = ['path_in',
                  'file_in',
                  'path_out'
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


    # Start
    print("________________________________________\n")
    print("               DRAWMAP")
    print("________________________________________\n")

    # Read input file
    inputs = read_inputs('drawmap.json')
    draw_map_1(inputs, 3)
    #draw_map_24(inputs)

def draw_map_1(inputs, n ):
    """draw 1 maps of a day"""
    draw_map = DrawMap(inputs)
    draw_map.read_head()

    draw_map.create_title(n)
    draw_map.reader_uv_by_time(n)
    print(draw_map.lats.shape, draw_map.lons.shape)
    print(draw_map.us.shape, draw_map.vs.shape)
    print(draw_map.title_full)
    draw_map.draw()

def draw_map_24(inputs):
    """draw 24+1 maps of a day"""
    draw_map = DrawMap(inputs)
    draw_map.read_head()

    for n in range(draw_map.reader.ini_ntime, 25+draw_map.reader.ini_ntime):
        draw_map.create_title(n)
        draw_map.reader_uv_by_time(n)
        print(draw_map.title_full)

        draw_map.draw()


class DrawMap:

    """Class to draw a map with all options"""

    def __init__(self, inputs):

        self.file_path_in = inputs['path_in']
        self.file_path_out = inputs['path_out']
        self.file_in = inputs['file_in']
        self.file_name = os.path.join(self.file_path_in, self.file_in)
        self.file_hdf_out = inputs['file_out']
        self.file_out = os.path.join(self.file_path_out, self.file_hdf_out)

        self.nx = inputs['nx']
        self.ny = inputs['ny']
        self.scale = inputs['scale']
        self.resolution = inputs['resolution']
        self.style = inputs['style']
        self.title = inputs['title']

        self.level = inputs['n_level']
        self.time = inputs['n_time']
        limits = inputs['limits']
        self.boundary_box = BoundaryBox(limits[0], limits[1], limits[2], limits[3])
        self.u_name = inputs['u']
        self.v_name = inputs['v']

        self.reader = None

    def read_head(self):
        print('Opening: {0}'.format(self.file_name))

        factory = read_factory(self.file_name)
        self.reader = factory.get_reader()

        with self.reader.open():

            lat = self.reader.latitudes
            lon = self.reader.longitudes

            if self.reader.coordinates_rank == 1:
                self.lats = lat[0:self.reader.n_latitudes - 2]  # ATENCIóN:Esto y lo de abajo era -1, revisar
                self.lons = lon[0:self.reader.n_longitudes - 2]
            elif self.reader.coordinates_rank == 2:
                self.lats = lat[0:self.reader.n_longitudes - 2, 0:self.reader.n_latitudes - 2]
                self.lons = lon[0:self.reader.n_longitudes - 2, 0:self.reader.n_latitudes - 2]

    def create_title(self, n_time):
        with self.reader.open():
            data = self.reader.get_date(n_time)
            data_str = data.strftime("%Y-%m-%d %H:%M UTC")
            data_comp = data.strftime("%Y%m%d%H%M")
            self.title_full = self.title + " " + data_str
            self.file_out_full = self.file_out + '_' + data_comp + '.png'

    def reader_uv_by_time(self, n_time):

        with self.reader.open():
            u = self.reader.get_variable(self.u_name, n_time)
            v = self.reader.get_variable(self.v_name, n_time)

            if len(u.shape) == 3:
                self.us = u[self.level, :-1, :- 1]
                self.vs = v[self.level, :-1, :-1]

            elif len(u.shape) == 2:
                self.us = u[:-1, :-1]
                self.vs = v[:-1, :-1]

            self.mod = pow((pow(self.us, 2) + pow(self.vs, 2)), .5)

    def reader_uv(self):
        self.reader_uv_by_time(self.time)

    def draw(self):
        drawcurrents(self.reader.coordinates_rank, self.nx, self.ny, self.scale, self.resolution,
                     self.level, self.time, self.lats, self.lons, self.us, self.vs, self.mod,
                     self.file_out_full, self.title_full, self.style, self.boundary_box)
        
if __name__ == '__main__':
    main()
