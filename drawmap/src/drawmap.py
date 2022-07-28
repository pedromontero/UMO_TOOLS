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
                  'limits',
                  'vector',
                  'scalar']
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
    if inputs['vector']:
        draw_map_vector(inputs, 3)
    if inputs['scalar']:
        draw_map_scalar(inputs, 3)
    #draw_map_24(inputs)

def draw_map_vector(inputs, n ):
    """draw 1 maps of a day"""
    draw_map = Vector(inputs)
    draw_map.read_head()
    draw_map.create_title(n)

    draw_map.reader_by_time()
    draw_map.draw()

def draw_map_scalar(inputs, n):
    draw_map = Scalar(inputs)
    draw_map.read_head()
    draw_map.create_title(n)

    draw_map.reader_by_time()
    draw_map.draw_scalar()

def draw_map_24(inputs):
    """draw 24+1 maps of a day"""
    draw_map = DrawMap(inputs)
    draw_map.read_head()

    for n in range(draw_map.reader.ini_ntime, 25+draw_map.reader.ini_ntime):
        draw_map.create_title(n)

        if draw_map.vector_bool:
            draw_map.vector.reader_by_time(draw_map)
            draw_map.vector.draw(draw_map)


class DrawMap:

    def __init__(self, inputs):
        self.options = OptionsMap(inputs)
        self.reader = self.get_reader(self.options.file_name)

    def read_head(self):

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
            self.title_full = self.options.title + " " + data_str
            self.file_out_full = self.options.file_out + '_' + data_comp + '.png'

    def get_reader(self, file_in):
        print('Opening: {0}'.format(file_in))
        factory = read_factory(file_in)
        return factory.get_reader()


class OptionsMap:

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

        self.vector_bool = inputs['vector']
        self.scalar_bool = inputs['scalar']


class Vector(DrawMap):

    def __init__(self, inputs):

        super().__init__(inputs)


        self.u_name = inputs['u']
        self.v_name = inputs['v']
        self.us = None
        self.vs = None
        self.modules = None


    def reader_by_time(self):

        with self.reader.open():
            u = self.reader.get_variable(self.u_name, self.options.time)
            v = self.reader.get_variable(self.v_name, self.options.time)

            if len(u.shape) == 3:
                self.us = u[self.options.level, :-1, :- 1]
                self.vs = v[self.options.level, :-1, :-1]

            elif len(u.shape) == 2:
                self.us = u[:-1, :-1]
                self.vs = v[:-1, :-1]

            self.modules = pow((pow(self.us, 2) + pow(self.vs, 2)), .5)

    def draw(self):
        drawcurrents(self.reader.coordinates_rank, self.options.nx, self.options.ny,
                     self.options.scale, self.options.resolution,
                     self.options.level, self.options.time,
                     self.lats, self.lons, self.us, self.vs, self.modules,
                     self.file_out_full, self.title_full, self.options.style,
                     self.options.boundary_box)


class Scalar(DrawMap):

    def __init(self, inputs):

        super().__init__(inputs)
        self.scalar_name = inputs['scalar_magnitude']
        self.scalars = None

    def reader_by_time(self):

        with self.reader.open():
            scalar = self.reader.get_variable(self.scalar_name, self.options.time)

            if len(scalar.shape) == 3:
                self.scalars = scalar[self.options.level, :-1, :- 1]

            elif len(scalar.shape) == 2:
                self.scalars = scalar[:-1, :-1]

    def draw_scalar(self):
        """

        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        dx = 0.01
        middle_lon = self.options.boundary_box.middle_lon()
        middle_lat =  self.options.boundary_box.middle_lat()
        m = Basemap(llcrnrlon=self.options.boundary_box.lon_min - dx,
                    llcrnrlat=self.options.boundary_box.lat_min - dx,
                    urcrnrlon= self.options.boundary_box.lon_max + dx,
                    urcrnrlat=self.options.boundary_box.lat_max + dx,
                    resolution=self.options.resolution, projection='tmerc', lon_0=middle_lon, lat_0=middle_lat)

        m.drawcoastlines()
        m.fillcontinents(color='grey', lake_color='aqua')

        # m.drawmapboundary(fill_color='aqua')

        if self.reader.coordinates_rank == 1:
            lon, lat = np.meshgrid(self.lons, self.lats)
        else:
            lon, lat = self.lons, self.lats

        x, y = m(lon, lat)

        # draw filled contours.
        clevs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        cs = m.contourf(x, y, self.scalars, clevs, cmap=plt.cm.jet)
        cbar = m.colorbar(cs, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('m/s')

        # add title
        plt.title(self.title_full)

        fig.savefig(self.file_out_full, dpi=100, facecolor='w', edgecolor='w', format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close(fig)

        return




        
if __name__ == '__main__':
    main()
