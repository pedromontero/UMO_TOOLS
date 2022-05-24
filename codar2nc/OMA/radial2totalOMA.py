#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from matplotlib.tri import Triangulation
from numpy.linalg import solve

from matplotlib import pyplot as plt

import xarray as xr

from glob import glob

# Importamos lo necesario para poder procesar radiales:
from radiales import Radial, Grid

deg2rad = np.pi/180
rad2deg = 1/deg2rad


def tsearch_arbitrary(p, t, x, y):

    # Triangulador y finder para localizar el triángulo:
    T = Triangulation(p[:,0],p[:,1],t)
    finder = T.get_trifinder()

    # Con los datos calculados por el finder:
    Tn = finder(x,y)

    # Coordenadas baricentricas:
    p1  = p[t[Tn][:,0]]
    v12 = p[t[Tn][:,1]] - p1
    v13 = p[t[Tn][:,2]] - p1
    dd  = v12[:,0]*v13[:,1] - v12[:,1]*v13[:,0]

    xx = x - p1[:,0]
    yy = y - p1[:,1]

    A12 = ( v13[:,1]*xx - v13[:,0]*yy )/dd
    A13 = ( v12[:,0]*yy - v12[:,1]*xx )/dd

    return Tn, A12, A13

if __name__ == '__main__':

    # Fichero con los modos precalculados:
    path = './midemo/codigo_matlab/mi_demo/misdatos/'
    path = './midemo/fit_OMA_modes_to_radials/'
    path = './datos/'
    m = scipy.io.loadmat('%s/modes.mat' % path,variable_names=['pLonLat', 't', 'ux_tri', 'uy_tri', 'border'],struct_as_record=True, squeeze_me=True)

    # Nodos triangulación resultante de pdetool
    p = m['pLonLat'].T

    # Triangulación resultante de pdetool (one-based):
    t  = m['t'][0:3].T
    t -= 1 # Pasamos a cero-based


    path = './datos/RadialFiles'
    prefijos = ['SILL','VILA','FIST','PRIO']
    prefijos = ['*']

    radiales = []
    for prefijo in prefijos:
        radiales += glob('%s/*_%s_2021_09_01_0000.ruv' % (path, prefijo)) # Solo una fecha de ejemplo
        # radiales += glob('%s/*_%s_2022_04_01_0400.ruv' % (path, prefijo))


    R = []

    for fichero in radiales:

        # Creamos el objeto radial para leer el fichero:
        radial  = Radial(fichero)

        # Creamos la malla donde queremos inscribir la tabla:
        grd = Grid(radial)

        # Metemos la tabla en la malla:
        radial.to_grid(grd)

        R.append(radial)

    ux, uy, Tn, head, speed, X, Y = [], [], [], [], [], [], []

    for r in R:

        # Nodos donde hay datos de radiales (hay 5 radiales, algunas sin datos. Cogemos la primera):
        x = r.variables['RDVA'].LONGITUDE.values.flatten()
        y = r.variables['RDVA'].LATITUDE.values.flatten()
    
        # Búsqueda de nodos y cálculo de coordenadas baricentricas:
        [tn,a12,a13] = tsearch_arbitrary( p, t, x, y )

        # Estas son las amplitudes de los modos en la malla radial (equivalente a toda pdeintrp_arbitrary en nuestro caso particular):
        ux.append( m['ux_tri'][tn,:] )
        uy.append( m['uy_tri'][tn,:] )

        # Otros acumuladores:
        X.append(x)
        Y.append(y)
        Tn.append(tn)

        # Esto es delicado. Aquí hay que ver si se usan las direcciones locales o el bearing. No tienen por qué coincidir si la malla es grande. Es decir, 
        # Un determinado bearing constante implica direcciones geográficas diferentes a lo largo del rayo. Esto no se notará a una escala de 10 km pero si
        # a una escala de 100 km.
        head.append(r.variables['DRVA'].values.squeeze().flatten())
        speed.append(r.variables['RDVA'].values.squeeze().flatten())

    # Reconstruimos las matrices con todos los datos a procesar:
    ux = np.row_stack(ux)
    uy = np.row_stack(uy)

    Tn = np.concatenate(Tn)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    head  = np.concatenate(head)

    # Cambio de convención para la dirección (meteorológica -> cartesiana):
    theta = np.arctan2(-np.cos(head*deg2rad), -np.sin(head*deg2rad))

    speed = np.concatenate(speed)

    # Ya en openMA_modes_fit_with_errors. Esto es la proyección de las amplitudes de los modos en la dirección del nodo, que no es otra cosa que 
    # el producto escalar de (ux,uy) por el vector unitario (cos(theta),sin(theta)):
    modes = (ux.T*np.cos(theta) + uy.T*np.sin(theta)).T

    # Resolución del sistema de ecuaciones:
    ## Filtrado de los nans en speed y puntos fuera del contorno:
    condicion = ~np.isnan(speed) & (Tn != -1)

    speed = speed[condicion]
    theta = theta[condicion]
    modes = modes[condicion]

    X  = X[condicion]
    Y  = Y[condicion]
    Tn = Tn[condicion]

    # Estos factores... no he investigado lo que son... Tampoco calculamos con matriz de pesos. 
    # q = modes.shape[0]
    # K = float(entrada['p']['K'])

    A = np.dot(modes.T,modes) # + K*q/2
    b = np.dot(modes.T,speed)

    alpha = solve(A,b)

    # Velocidades totales interpoladas en la malla triangular:
    Ux = (m['ux_tri']*alpha).sum(axis=1)
    Uy = (m['uy_tri']*alpha).sum(axis=1)

    # Interpolación a una malla regular por el método de NN:

    ## Triangulador:
    T = Triangulation(p[:,0],p[:,1],t)
    finder = T.get_trifinder()

    ## Necesitamos especificar una malla de salida. En este caso cogemos un fichero del THREDDS:
    malla = xr.open_dataset('./datos/HFR-Galicia-Total_2022_05_14_0000.nc')

    #np.meshgrid(malla.lon,malla.lat)

    lon, lat = np.meshgrid(malla.LONGITUDE, malla.LATITUDE)
    asignacion = finder(lon,lat)

    Tx = Ux[asignacion]
    Ty = Uy[asignacion]

    Tx[asignacion==-1] = np.nan
    Ty[asignacion==-1] = np.nan

    ##############
    # RESULTADOS #
    ##############

    # AUX: Coordendas de los baricentros de la triangulación original:
    BX = (p[:,0][t]/3).sum(axis=1)
    BY = (p[:,1][t]/3).sum(axis=1)

    # En la malla triangular:
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    ax1.triplot(T, 'k-', lw=1)

    # Resultado de la interpolación:
    ## Módulo de la interpolación en la malla triangular:
    clb = ax1.tripcolor(T,np.sqrt(Ux*Ux+Uy*Uy))
    plt.colorbar(clb)

    '''
    ##  Vectores resultado de la interpolación:
    ax1.quiver(BX,BY,Ux,Uy,scale=10)

    ## Vectores resultado de la interpolación solo en los triangulos con dato de radial:
    ax1.quiver(BX[Tn],BY[Tn],Ux[Tn],Uy[Tn],scale=10)
    '''

    ## Componente radial del resultado de la interpolación:
    modulo = Ux[Tn]*np.cos(theta) + Uy[Tn]*np.sin(theta)
    ax1.quiver(BX[Tn],BY[Tn],modulo*np.cos(theta),modulo*np.sin(theta),scale=10)

    ## Radiales usadas en la inteprolación:
    ax1.quiver(X, Y, speed*np.cos(theta), speed*np.sin(theta),color='r',scale=10)

    plt.grid()

    plt.show()


    fig1, ax1 = plt.subplots()

    ax1.quiver(malla.LONGITUDE, malla.LATITUDE, malla.EWCT.squeeze(), malla.NSCT.squeeze(),scale=10)
    ax1.quiver(BX,BY,Ux,Uy,scale=10,color='r')
    ax1.plot(X,Y,'b.')

    ax1.set_aspect('equal')

    plt.grid()

    plt.show()


    fig1, ax1 = plt.subplots()
    #ax1.pcolor(malla.lon,malla.lat,np.sqrt(Tx**2+Ty**2),vmin=0,vmax=0.5)
    ax1.pcolor(malla.LONGITUDE,malla.LATITUDE,Ty,vmin=-0.5,vmax=0.5)
    ax1.quiver(malla.LONGITUDE,malla.LATITUDE,Tx, Ty, scale=10)
    plt.grid()
    ax1.set_aspect('equal')
    a = ax1.axis()

    fig1, ax1 = plt.subplots()
    #ax1.pcolor(malla.lon,malla.lat,np.sqrt(malla.u**2+malla.v**2).squeeze(),vmin=0,vmax=0.5)
    ax1.pcolor(malla.LONGITUDE,malla.LATITUDE,malla.NSCT.squeeze(),vmin=-0.5,vmax=0.5)

    ax1.quiver(malla.LONGITUDE,malla.LATITUDE,malla.NSCT.squeeze(), malla.EWCT.squeeze(), scale=10)
    plt.grid()
    ax1.set_aspect('equal')
    ax1.axis(a)

    plt.show()

