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


from customize import add_lib
add_lib()

from intecmar.fichero import input_file


def drawcurrents(nx, ny, scale, resolution, level, time, lats, lons, ust, vst, mod, file_out, title, style, limits):
    """

    :return:
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    m = Basemap(llcrnrlon=limits[0], llcrnrlat=limits[1], urcrnrlon=limits[2], urcrnrlat=limits[3],
                resolution=resolution, projection='tmerc', lon_0=-8, lat_0=42)

    m.drawcoastlines()
    m.fillcontinents(color='grey', lake_color='aqua')


    #m.drawmapboundary(fill_color='aqua')

    lon, lat = np.meshgrid(lons, lats)
    x, y = m(lon, lat)

    # draw filled contours.
    clevs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    if style == 'contour':
        cs = m.contourf(x, y, mod, clevs, cmap=plt.cm.jet)
        plt.quiver(x[::nx, ::ny], y[::nx, ::ny], ust[::nx, ::ny], vst[::nx, ::ny], scale=scale)

    if style == 'cvector':
        clim = [0., 0.5]
        cs = plt.quiver(x[::nx, ::ny], y[::nx, ::ny], ust[::nx, ::ny], vst[::nx, ::ny], mod[::nx, ::ny], clim=clim, scale=scale)

    if style == 'stream':

        print(lons.shape)
        print(x[::, 1])
        print(ust.shape)
        xx = x[1::, ::]
        xxx = xx[0]
        yyy = y[::, 1::][1]

        u = ust[::, ::]
        v = vst[::, ::]
        print(len(xxx))
        print(len(yyy))
        print(len(u))
        print(len(v))

        cs = plt.streamplot(lon, lat, u, v)

    cbar = m.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('m/s')

    # add title
    plt.title(title)

    fig.savefig(file_out, dpi=100, facecolor='w', edgecolor='w', format='png',
                transparent=False,  bbox_inches='tight', pad_inches=0.1)
    plt.clf()

    return


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

    # Read file with input keywords
    input_nome = 'drawmap.dat'  # Files must exist with keywords below
    input_chaves = ['PATH', 'FILEIN', 'NX', 'NY', 'RESOLUTION', 'SCALE', 'NTIME', 'NLEVEL', 'FILEOUT', 'TITLE', 'STYLE',
                    'LIMITS']
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
    style = input_retorno[10].strip()
    limits_str = input_retorno[11].split(';')
    limits = []
    for i in range(len(limits_str)):
        limits.append(float(limits_str[i]))
    print('Opening: {0}'.format(file_name))

    f = h5py.File(file_name,  "r")

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

    u = f['/Results/velocity U/velocity U_0000'+time]
    v = f['/Results/velocity V/velocity V_0000'+time]

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

    drawcurrents(nx, ny, scale, resolution, level, time, lats, lons, ust, vst, mod, file_out, title, style, limits)


if __name__ == '__main__':
    main()







