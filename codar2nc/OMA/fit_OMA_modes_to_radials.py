#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_OMA_modes_to_radials.py

@Purpose:
@version: 0.0

@python version: 3.9
@author: Pedro Montero
@license:
@requires:

@date

@history:

"""
import scipy.io

mat = scipy.io.loadmat('modes07.mat')

for key in mat.keys():
    print(key)
    print('----------------')
    #print(mat[key])
    print('\n')
