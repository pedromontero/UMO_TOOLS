
from cartopy.geodesic import Geodesic

import pyproj
import numpy as np


g = Geodesic()


site_lon = [-9.2721333, -8.89715]
site_lat = [42.8820833,42.1045]

puntos = [[-9.9296003, 40.6584154],[-9.8586619, 40.6582594], [-9.787724, 40.6580589]]

resultado = [
[-167.29935471, -168.63466007, -169.98191874],
[-151.38889893, -153.08225395, -154.82606582]
 ]

def ginverse(coord_in, puntos):
    g = pyproj.Geod(ellps="WGS84")
    lon_in, lat_in = coord_in
    output = []
    for punto in puntos:
        output.append((g.inv(lon_in, lat_in, punto[0], punto[1])))
    return np.array(output)

for i in range(0, 2):
    lon = site_lon[i]

    lat = site_lat[i]
    ang = g.inverse((lon, lat), puntos)[:, 1]
    print(i, lon, lat, ang)

gg = pyproj.Geod(ellps="WGS84")

for i in range(0, 2):
    lon = site_lon[i]
    lat = site_lat[i]

    ang = ginverse((lon, lat), puntos)[:, 0]
    print(i, lon, lat, ang)

