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
import sys
import os


def input_file(nome, chaves):
    """Manages input keywords file.

    :param nome: input file with keywords
    :param chaves: list of keywords
    :return: a string of the value of keyword

    Keywords will be separated from values with two double colon and one space:  ':: '
    example (keyword: foo)
    foo:: value

    The value could be a one line string or a multiline string. In the last case, multiline string starts
    with '<<<' and ends with '>>>'.
    example:
    foo:: <<< This is a multiline string
              ending in this line>>>

    If the value is a list, it will be also processed as string and it should be convert to a list out of the function.
    ';' is suggested to separate the elements of a list.
     """

    retorno = []
    if os.path.isfile(nome):
        print("I am opening ", nome)
        for chave in chaves:
            chave_found = False
            f = open(nome, "r")
            linea = f.readline()
            while linea != "":
                if linea.find(":: ") >= 0:
                    trozos = linea.split(':: ')
                    chave_in = trozos[0].strip()
                    retorno_in = trozos[1].strip()
                    if chave_in == chave:
                        chave_found = True
                        if retorno_in.find("<<<") >= 0:
                            retorno_in = retorno_in.split("<<<")[1]
                            while retorno_in.find(">>>") < 0:
                                nova = f.readline()
                                retorno_in += nova
                            retorno_in = retorno_in.split(">>>")[0]
                        retorno.append(retorno_in)
                linea = f.readline()
            if not chave_found:  # If keyword is not present, return None
                retorno.append(None)
            f.close()
        return retorno

    else:
        print("A file: " + nome + " is needed")
        sys.exit()


def drawcurrents(nx, ny, scale, resolution, level, time, lats, lons, ust, vst, mod, file_out, title):
    """

    :return:
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    m = Basemap(llcrnrlon=-9.1, llcrnrlat=42.42, urcrnrlon=-8.76, urcrnrlat=42.67,
                resolution=resolution, projection='tmerc', lon_0=-8, lat_0=42)

    m.drawcoastlines()
    m.fillcontinents(color='grey', lake_color='aqua')


    #m.drawmapboundary(fill_color='aqua')

    lon, lat = np.meshgrid(lons, lats)
    x, y = m(lon, lat)

    # draw filled contours.
    clevs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cs = m.contourf(x, y, mod, clevs, cmap=plt.cm.jet)

    # add colorbar.
    cbar = m.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label('m/s')

    # add title
    plt.title(title)

    plt.quiver(x[::nx, ::ny], y[::nx, ::ny], ust[::nx, ::ny], vst[::nx, ::ny], scale=scale)
    '''fig.savefig(file_out, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)'''
    fig.savefig(file_out, dpi=300, facecolor='w', edgecolor='w', format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.clf()

    return


def main():
    """
    Main program:


    :return:
    """

    # Start
    print("________________________________________\n")
    print("               DRAWMAP")
    print("________________________________________\n")

    # Read file with input keywords
    input_nome = 'drawmap.dat'  # Files must exist with keywords below
    input_chaves = ['PATH', 'FILEIN', 'NX', 'NY', 'RESOLUTION', 'SCALE', 'NTIME', 'NLEVEL', 'FILEOUT', 'TITLE']
    input_retorno = input_file(input_nome, input_chaves)

    file_path = input_retorno[0].strip()
    file_hdf = input_retorno[1].strip()
    nx = int(input_retorno[2].strip())
    ny = int(input_retorno[3].strip())
    scale = int(input_retorno[5].strip())
    resolution = input_retorno[4].strip()
    level = int(input_retorno[7].strip())
    time = input_retorno[6].strip()
    file_hdf_out = input_retorno[8].strip()
    file_out = os.path.join(file_path, file_hdf_out)
    title = input_retorno[9].strip()
    file_name = os.path.join(file_path, file_hdf)
    print('Opening: {0}'.format(file_name))

    f = h5py.File(file_name,  "r")

    lat = f['/Grid/Latitude']
    lon = f['/Grid/Longitude']
    u = f['/Results/velocity U/velocity U_000'+time]
    v = f['/Results/velocity V/velocity V_000'+time]

    nlon = lat.shape[0]
    nlat = lon.shape[1]

    lats = lat[0, 0:nlat-1]
    lons = lon[0:nlon-1, 0]

    us = u[level, 0:nlon-1, 0:nlat-1]
    vs = v[level, 0:nlon-1, 0:nlat-1]
    ust = np.transpose(us)
    vst = np.transpose(vs)

    mod = pow((pow(ust, 2) + pow(vst, 2)), .5)

    f.close()

    drawcurrents(nx, ny, scale, resolution, level, time, lats, lons, ust, vst, mod, file_out, title)


if __name__ == '__main__':
    main()







