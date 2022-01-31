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

def read_inputs(input_file):
    """Read keywords for options"""
    input_keys = ['path_in',
                  'file_in',
                  'path_out'
                  'file_out'
                  ]
    return read_input(input_file, input_keys)


def main():
    """
    Main program:


    :return:
    """


    # Start
    print("________________________________________\n")
    print("               UI")
    print("________________________________________\n")

    # Read input file
    inputs = read_inputs('ui.json')
    ui = UI(inputs)



class UI:

    """Class to draw a map with all options"""

    def __init__(self, inputs):

        self.file_path_in = inputs['path_in']
        self.file_path_out = inputs['path_out']
        self.file_in = inputs['file_in']
        self.file_name = os.path.join(self.file_path_in, self.file_in)
        self.file_out = inputs['file_out']
        self.file_out = os.path.join(self.file_path_out, self.file_out)
        self.reader = None

    def read_head(self):
        print('Opening: {0}'.format(self.file_name))

        factory = read_factory(self.file_name)
        self.reader = factory.get_reader()

        with self.reader.open():

            lat = self.reader.latitudes
            lon = self.reader.longitudes

            if self.reader.coordinates_rank == 1:
                self.lats = lat[0:self.reader.n_latitudes - 1]
                self.lons = lon[0:self.reader.n_longitudes - 1]
            elif self.reader.coordinates_rank == 2:
                self.lats = lat[0:self.reader.n_longitudes - 1, 0:self.reader.n_latitudes - 1]
                self.lons = lon[0:self.reader.n_longitudes - 1, 0:self.reader.n_latitudes - 1]

    def reader_uv(self):

        with self.reader.open():
            u = self.reader.get_variable(self.u_name, 0)
            v = self.reader.get_variable(self.v_name, 0)

            if len(u.shape) == 3:
                self.us = u[self.level, :-1, :- 1]
                self.vs = v[self.level, :-1, :-1]

            elif len(u.shape) == 2:
                self.us = u[:-1, :-1]
                self.vs = v[:-1, :-1]

            self.mod = pow((pow(self.us, 2) + pow(self.vs, 2)), .5)

