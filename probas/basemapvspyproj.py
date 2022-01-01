

from mpl_toolkits.basemap import Basemap

import pyproj
import numpy as np
import numpy.ma as ma


def rotate_vector(pr,uin, vin, lons, lats, returnxy=False):
    """
    Rotate a vector field (``uin,vin``) on a rectilinear grid
    with longitudes = ``lons`` and latitudes = ``lats`` from
    geographical (lat/lon) into map projection (x/y) coordinates.
    Differs from transform_vector in that no interpolation is done.
    The vector is returned on the same grid, but rotated into
    x,y coordinates.
    The input vector field is defined in spherical coordinates (it
    has eastward and northward components) while the output
    vector field is rotated to map projection coordinates (relative
    to x and y). The magnitude of the vector is preserved.
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    uin, vin         input vector field on a lat/lon grid.
    lons, lats       Arrays containing longitudes and latitudes
                     (in degrees) of input data in increasing order.
                     For non-cylindrical projections (those other than
                     ``cyl``, ``merc``, ``cyl``, ``gall`` and ``mill``) lons
                     must fit within range -180 to 180.
    ==============   ====================================================
    Returns ``uout, vout`` (rotated vector field).
    If the optional keyword argument
    ``returnxy`` is True (default is False),
    returns ``uout,vout,x,y`` (where ``x,y`` are the map projection
    coordinates of the grid defined by ``lons,lats``).
    """
    # if lons,lats are 1d and uin,vin are 2d, and
    # lats describes 1st dim of uin,vin, and
    # lons describes 2nd dim of uin,vin, make lons,lats 2d
    # with meshgrid.
    if lons.ndim == lats.ndim == 1 and uin.ndim == vin.ndim == 2 and \
            uin.shape[1] == vin.shape[1] == lons.shape[0] and \
            uin.shape[0] == vin.shape[0] == lats.shape[0]:
        lons, lats = np.meshgrid(lons, lats)
    else:
        if not lons.shape == lats.shape == uin.shape == vin.shape:
            raise TypeError("shapes of lons,lats and uin,vin don't match")
    x, y = pr(lons, lats)

    # rotate from geographic to map coordinates.
    if ma.isMaskedArray(uin):
        mask = ma.getmaskarray(uin)
        masked = True
        uin = uin.filled(1)
        vin = vin.filled(1)
    else:
        masked = False

    # Map the (lon, lat) vector in the complex plane.
    uvc = uin + 1j * vin
    uvmag = np.abs(uvc)
    theta = np.angle(uvc)

    # Define a displacement (dlon, dlat) that moves all
    # positions (lons, lats) a small distance in the
    # direction of the original vector.
    dc = 1E-5 * np.exp(theta * 1j)
    dlat = dc.imag * np.cos(np.radians(lats))
    dlon = dc.real

    # Deal with displacements that overshoot the North or South Pole.
    farnorth = np.abs(lats + dlat) >= 90.0
    somenorth = farnorth.any()
    if somenorth:
        dlon[farnorth] *= -1.0
        dlat[farnorth] *= -1.0

    # Add displacement to original location and find the native coordinates.
    lon1 = lons + dlon
    lat1 = lats + dlat
    xn, yn = pr(lon1, lat1)

    # Determine the angle of the displacement in the native coordinates.
    vecangle = np.arctan2(yn - y, xn - x)
    if somenorth:
        vecangle[farnorth] += np.pi

    # Compute the x-y components of the original vector.
    uvcout = uvmag * np.exp(1j * vecangle)
    uout = uvcout.real
    vout = uvcout.imag

    if masked:
        uout = ma.array(uout, mask=mask)
        vout = ma.array(vout, mask=mask)
    if returnxy:
        return uout, vout, x, y
    else:
        return uout, vout


def main():

    origen_lon = -8.
    origen_lat = 42.
    print(' Código en el programa:')
    # Escogemos una proyección. Tmercator está bien. La idea es trabajar en un plano:
    # m = Basemap(llcrnrlon=-11.0, llcrnrlat=41.8, urcrnrlon=-8, urcrnrlat=44.5, resolution='h', projection='tmerc', lon_0=-8, lat_0=45)
    m = Basemap(llcrnrlon=-11.0, llcrnrlat=41.8, urcrnrlon=-8, urcrnrlat=44.5, resolution='l', projection='tmerc',
                lon_0=origen_lon, lat_0=origen_lat)

    print(f'proj usado por basemap: {m.proj4string}')

    print('\nUsando basemap:')
    print(f'Coordenadas iniciales: {origen_lat}, {origen_lon}')
    # Necesito las coordenadas del origen y su proyección:
    origen_x, origen_y = m(origen_lon, origen_lat)
    print(f'origen_x : {origen_x}, origen_y: {origen_y}')

    print('\nUsando pyproj:')

    pr = pyproj.Proj('+proj=tmerc +bR_a=6370997.0 +units=m +lat_0=42.0 +lon_0=-8.0 +x_0=249341.9581021159 +y_0=17861.19187674373 ')
    origen_x, origen_y = pr(origen_lon, origen_lat)
    print(f'origen_x : {origen_x}, origen_y: {origen_y}')

    # recuperar las coordenadas
    print('recuperando las coordenadas iniciales:')
    print(pr(origen_x, origen_y, inverse=True))

    print('\nrotate:')

    lat = np.array([42,41.5,42,42.5])
    lon = np.array([-8.,-8.5,-9,-9.5])
    u = np.array([1,1,-1,0])
    v = np.array([0,0,0,-1])

    X, Y = m.rotate_vector(u, v, lon, lat )
    print('Usando basemap:')
    print(X,Y)
    print('Usando pyproj:')
    X, Y = rotate_vector(pr,u, v, lon, lat)
    print(X,Y)


if __name__=='__main__':
    main()



