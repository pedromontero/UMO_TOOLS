
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


def inverse(points, endpoints):
    """
    Solve the inverse geodesic problem.

    Can accept and broadcast length 1 arguments. For example, given a
    single start point, an array of different endpoints can be supplied to
    find multiple distances.

    Parameters
    ----------
    points: array_like, shape=(n *or* 1, 2)
        The starting longitude-latitude point(s) from which to travel.

    endpoints: array_like, shape=(n *or* 1, 2)
        The longitude-latitude point(s) to travel to.

    Returns
    -------
    `numpy.ndarray`, shape=(n, 3)
        The distances, and the (forward) azimuths of the start and end
        points.

    """
    geod = pyproj.Geod(ellps="WGS84")

    # Create numpy arrays from inputs, and ensure correct shape.
    points = np.array(points, dtype=np.float64)
    endpoints = np.array(endpoints, dtype=np.float64)

    if points.ndim > 2 or (points.ndim == 2 and points.shape[1] != 2):
        raise ValueError(
            f'Expecting input points to be (N, 2), got {points.shape}')

    pts = points.reshape((-1, 2))
    epts = endpoints.reshape((-1, 2))

    sizes = [pts.shape[0], epts.shape[0]]
    n_points = max(sizes)
    if not all(size in [1, n_points] for size in sizes):
        raise ValueError("Inputs must have common length n or length one.")

    # Broadcast any length 1 arrays to the correct size.
    if pts.shape[0] == 1:
        orig_pts = pts
        pts = np.empty([n_points, 2], dtype=np.float64)
        pts[:, :] = orig_pts

    if epts.shape[0] == 1:
        orig_pts = epts
        epts = np.empty([n_points, 2], dtype=np.float64)
        epts[:, :] = orig_pts

    start_azims, end_azims, dists = geod.inv(pts[:, 0], pts[:, 1],
                                                  epts[:, 0], epts[:, 1])
    # Convert back azimuth to forward azimuth.
    end_azims += np.where(end_azims > 0, -180, 180)
    return np.column_stack([dists, start_azims, end_azims])


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

for i in range(0, 2):
    lon = site_lon[i]

    lat = site_lat[i]
    ang = inverse((lon, lat), puntos)[:, 1]
    print(i, lon, lat, ang)

