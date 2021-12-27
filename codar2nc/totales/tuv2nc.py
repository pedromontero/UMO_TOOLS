# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import xarray as xr

import re

from collections import OrderedDict

from datetime import datetime, timedelta

from cartopy.geodesic import Geodesic

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree, KDTree

import argparse

from glob import glob

import json

import os

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

deg2rad = np.pi/180 

dtypes        = {"TIME"        : 'float64',
                 "DEPH"        : 'float32',
                 "BEAR"        : 'float32',
                 "RNGE"        : 'float32',
                 "LONGITUDE"   : 'float32',
                 "LATITUDE"    : 'float32',
                 "EWCT"        : 'int16',
                 "NSCT"        : 'int16',
                 "EWCS"        : 'int16',
                 "NSCS"        : 'int16',
                 "GDOP"        : 'int16',
                 "CCOV"        : 'int32',
                 "DDENS"       : 'int32', # Variable auxiliar. No saldrá en el netCDF
                 "err"         : 'int8',  # Variable auxiliar. No saldrá en el netCDF
                 "NARX"        : 'int8',
                 "NATX"        : 'int8',
                 "SLTR"        : 'int32',
                 "SLNR"        : 'int32',
                 "SLTT"        : 'int32',
                 "SLNT"        : 'int32',
                 "TIME_QC"     : 'int8',
                 "POSITION_QC" : 'int8',
                 "DEPH_QC"     : 'int8',
                 "QCflag"      : 'int8',
                 "VART_QC"     : 'int8',
                 "GDOP_QC"     : 'int8',
                 "CSPD_QC"     : 'int8',
                 "DDNS_QC"     : 'int8'}

scale_factors = {"XDST"        : 0.001,
                 "YDST"        : 0.001,
                 "EWCT"        : 0.001,
                 "NSCT"        : 0.001,
                 "EWCS"        : 0.001,
                 "NSCS"        : 0.001,
                 "GDOP"        : 0.001,
                 "CCOV"        : 1E-06,
                 "DDENS"       : 1, # Variable auxiliar. No saldrá en el netCDF
                 "err"         : 1, # Variable auxiliar. No saldrá en el netCDF
                 "NARX"        : 1,
                 "NATX"        : 1,
                 "SLTR"        : 0.001,
                 "SLNR"        : 0.001,
                 "SLTT"        : 0.001,
                 "SLNT"        : 0.001,
                 "TIME_QC"     : 1,
                 "POSITION_QC" : 1,
                 "DEPH_QC"     : 1,
                 "QCflag"      : 1,
                 "VART_QC"     : 1,
                 "GDOP_QC"     : 1,
                 "CSPD_QC"     : 1,
                 "DDNS_QC"     : 1}

add_offsets = {}

for key, value in scale_factors.items():
    if isinstance(value, float):

        scale_factors[key] = np.float32(scale_factors[key])
        add_offsets[key]   = np.float32(0)

    else:

        # Generamos un conversor de tipo a partir del tipo de la variable:
        conversor           = np.dtype(dtypes[key])
    
        # Utilizamos el conversor para recodificar un tipo nativo de python a un escalar tipo numpy:
        scale_factors[key] = np.int_(scale_factors[key]).astype(conversor)
        add_offsets[key]   = np.int_(0).astype(conversor)


_FillValues   = {}

for key, value in dtypes.items():
    if 'float' in value:
        _FillValues[key] = np.finfo(dtypes[key]).min+1
    else:
        _FillValues[key] = np.iinfo(dtypes[key]).min+1

class Total():

    """
    Clase de abstracción para la lectura y procesamiento de ficheros totales (.tuv)

    Atributos
    ---------

    Metodos
    -------
    """

    def __init__(self, fichero):

        """
        Constructor

        Parametros
        ----------
        fichero: Fichero .tuv con las velocidades radiales
        """

        # El archivo tiene que ser abierto como binary:
        contenido = [linea.decode('utf-8').replace('%','').replace('\n','') for linea in  open(fichero,'rb').readlines() if '%%' not in  str(linea)]


        metadatos = [linea for linea in contenido if 'Table' not in linea]
        metadatos = dict([(linea.split(':')[0],linea.split(':')[1]) for linea in metadatos if ':' in str(linea) ])

        # Parseamos algunos metadatos que necesitaremos:
        self.GridSpacing              = 6.0
        self.TimeStamp                = datetime.strptime(metadatos['TimeStamp'],' %Y %m %d %H %M %S')

        # Líneas inicial y final de las tablas:
        starts  = np.arange(len(contenido))[['TableStart' in linea for linea in contenido]]
        ends    = np.arange(len(contenido))[['TableEnd' in linea for linea in contenido]]
        lengths = ends - starts - 1

        # Linea que contiene el header:
        columns = np.arange(len(contenido))[['TableColumnTypes' in linea for linea in contenido]]

        tablas = []

        # Aquí podemos aplicar los cambios en los nombres de las variables:
        headers    = [contenido[indice].split(':')[1].split() for indice in columns]

        # Solo estas columnas son fijas:
        header_comun = ['LOND','LATD','EWCT','NSCT','OWTR_QC','EWCS','NSCS','CCOV','XDST','YDST','RNGE','BEAR','RDVA','DRVA']
        ## Originales:   LOND   LATD   VELU   VELV   VFLG      UQAL   VQAL   CQAL   XDST   YDST   RNGE   BEAR   VELO   HEAD

        headers[0][0:14] = header_comun

        for i in range(2):

            if lengths[i] != 0:

                start = starts[i] + 1
                end   = ends[i]

                tablas.append(pd.DataFrame(np.array([linea.replace('"','').split() for linea in contenido[start:end]]), columns=headers[i]))


        tipos = {'LOND': np.dtype(float),
                 'LATD': np.dtype(float),
                 'EWCT': np.dtype(float),
                 'NSCT': np.dtype(float),
                 'OWTR_QC': np.dtype(int),
                 'EWCS': np.dtype(float),
                 'NSCS': np.dtype(float),
                 'CCOV': np.dtype(float),
                 'XDST': np.dtype(float),
                 'YDST': np.dtype(float),
                 'RNGE': np.dtype(float),
                 'BEAR': np.dtype(float),
                 'RDVA': np.dtype(float),
                 'DRVA': np.dtype(float)}

        # Hay que gestionar el número variable de columnas del final:
        for i in range(1,len(headers[0]) - len(header_comun) + 1):
            tipos['S%iCN' % i] = np.dtype(int)

        tablas[0] = tablas[0].astype(tipos)

        tipos = {'SNDX': np.dtype(int),
                 'SITE': np.dtype('U4'),
                 'OLAT': np.dtype(float),
                 'OLON': np.dtype(float),
                 'COVH': np.dtype(float),
                 'RNGS': np.dtype(float),
                 'PATK': np.dtype('U4'),
                 'REFB': np.dtype(float),
                 'NUMV': np.dtype(float),
                 'MAXN': np.dtype(float),
                 'MAXS': np.dtype(float),
                 'MAXE': np.dtype(float),
                 'MAXW': np.dtype(float),
                 'PATH': np.dtype('U72'),
                 'UUID': np.dtype('U36')}

        tablas[1] = tablas[1].astype(tipos)

        # Factores de conversión necesarios para algunas columnas de la tabla:
        tablas[0].EWCT /=  100.
        tablas[0].NSCT /=  100.
        tablas[0].EWCS  = 0.01*np.sqrt(tablas[0].EWCS)
        tablas[0].NSCS  = 0.01*np.sqrt(tablas[0].NSCS)
        tablas[0].CCOV *= 0.0001

        # Variables derivadas de columnas de las tablas:

        ## DDENS depende del número de radares activos:
        tablas[0]['DDENS'] = tablas[0][['S%iCN' % i for i in tablas[1].SNDX]].sum(axis=1).astype(float)

        self.radares   = tablas[1].SITE.values
        print(self.radares)

        ## Objeto para calcular los ángulos geodésicos:
        self.g = Geodesic() # Por defecto, cálculos en WGS84 como elipsoide de referencia
        print(self.g)

        puntos  = np.column_stack([tablas[0].LOND.values, tablas[0].LATD.values])

        self.radialAngles = np.empty((len(tablas[0]),len(self.radares)))
        self.max_ortho    = np.empty((len(tablas[0]),2))

        for i, (SNDX, siteLon, siteLat) in enumerate(tablas[1][['SNDX', 'OLON', 'OLAT']].values):
                print(i,SNDX,siteLon, siteLat, self.radialAngles[:,i])
                print(self.g.inverse((siteLon, siteLat), puntos)[:,1])
        # Usamos el objerto de Cartopy para calcular los ángulos:
        for i, (SNDX, siteLon, siteLat) in enumerate(tablas[1][['SNDX','OLON','OLAT']].values):
            #self.radialAngles[:,i] = self.g.inverse((siteLon, siteLat), puntos).base[:,1]
            self.radialAngles[:, i] = self.g.inverse((siteLon, siteLat), puntos)[:, 1]

        # Angulos en el rango [0,360]
        self.radialAngles += 360
        self.radialAngles %= 360

        # Filtramos los ángulos y nos quedamos con los dos angulos cuya ortogonalidad es máxima:
        X, Y = np.meshgrid(range(len(self.radares)),range(len(self.radares)))

        for i,angulos in enumerate(self.radialAngles):

            c = np.cos(np.array([angulos])*deg2rad)
            s = np.sin(np.array([angulos])*deg2rad)

            t = c*c.T + s*s.T

            self.max_ortho[i] = angulos[ X[t==t.min()] ]

        GDOP = np.empty(len(tablas[0]))

        # Calculamos el GDOP:
        for i,angulos in enumerate(self.max_ortho):

            A = np.array([np.cos(deg2rad*angulos),np.sin(deg2rad*angulos)])
            GDOP[i] = np.sqrt( np.linalg.inv(np.dot(A,A.T)).trace() )

        tablas[0]['GDOP'] = GDOP

        # Clean Totals:
        err  = np.sqrt(tablas[0].EWCT**2+tablas[0].NSCT**2) > Total_QC_params.VelThr # Valores posibles del error [0,1]
        err += 2*(GDOP > Total_QC_params.GDOPThr)                                    # Valores posibles del error [0,2]
        # Por lo tanto este codigo de error puede tener los valores [0,1,2,3]

        tablas[0]['err'] = err.astype('float')

        self.metadatos = metadatos
        self.tablas    = tablas

    def to_grid(self, grid):

        longitud, latitud = np.meshgrid(grid.longitud, grid.latitud)

        destino = np.column_stack([longitud.flatten(), latitud.flatten()])
        puntos  = np.column_stack([self.tablas[0].LOND.values, self.tablas[0].LATD.values])

        # Busqueda cKDTree:
        nearest = cKDTree(puntos)
        distancias, vecinos = nearest.query(destino)

        # Generamos la máscara con las distancias:
        mascara = np.ones_like(longitud)
        condicion = distancias.reshape(longitud.shape)>0.05
        mascara[condicion] = np.nan

        variables = ['EWCT', 'NSCT', 'EWCS', 'NSCS', 'CCOV', 'DDENS', 'GDOP', 'err']

        self.variables = OrderedDict()

        # Necesitamos completar la lista de coordenadas:
        delta = self.TimeStamp - datetime(1950,1,1)
        self.variables['TIME'] = xr.DataArray([delta.days + delta.seconds/86400], dims   = {'TIME' : 1})
        self.variables['DEPH'] = xr.DataArray([0.], dims   = {'DEPTH' : 1})

        for variable in variables:
            
            # Creamos la matriz que se llenará con los datos:
            tmp          = np.ones_like(longitud.flatten())*np.nan

            # Asignamos los vecinos más próximos:
            tmp = self.tablas[0][variable][vecinos].values

            # Volvemos a la forma original:
            tmp  = tmp.reshape(longitud.shape)
            tmp *= mascara

            # Creamos el DataArray:
            if variable in ['EWCT', 'NSCT', 'EWCS', 'NSCS', 'CCOV', 'DDENS', 'GDOP', 'err']:
                # Crecemos en DEPTH:
                tmp = np.expand_dims(tmp,axis=0)

                # Crecemos en TIME:
                tmp = np.expand_dims(tmp,axis=0)

                self.variables[variable] = xr.DataArray(tmp, 
                                           dims   = {'TIME' : 1, 'DEPTH' : 1, 'LATITUDE' : grid.nLAT, 'LONGITUDE' : grid.nLON},
                                           coords = {'TIME' : self.variables['TIME'], 'DEPH' : self.variables['DEPH'], 'LATITUDE' : grid.latitud, 'LONGITUDE' : grid.longitud})

                # Encoding de las variables en el fichero:
                self.variables[variable].encoding["scale_factor"] = scale_factors[variable]
                self.variables[variable].encoding["add_offset"  ] = add_offsets[variable]
                self.variables[variable].encoding["dtype"       ] = dtypes[variable]
                self.variables[variable].encoding["_FillValue"  ] = _FillValues[variable]

    def QC_control(self):

        """
        Método para el control de calidad de los datos

        Parametros
        ----------
        Utiliza la lista de variables del objeto, que deben estar ya ongrid.
        """

        U_grid = self.variables['EWCT']
        V_grid = self.variables['NSCT']
        # U_std  = self.variables['EWCS']
        # V_std  = self.variables['NSCS']
        DDENS  = self.variables['DDENS']

        # Construimos alias de las variables para no tener que reescribir mucho código:
        
        # Variance Threshold QC test
        # variance = ((mat_tot.U_grid.^2).*(mat_tot.U_std).^2) + ((mat_tot.V_grid.^2).*(mat_tot.V_std).^2);
        # variance = ((U_grid**2)*(U_std**2) + (V_grid**2)*(V_std)**2)
    
        # Velocity Threshold QC test
        # totVel = sqrt(((mat_tot.U_grid).^2) + ((mat_tot.V_grid).^2));
        totVel = np.sqrt(U_grid**2 + V_grid**2)

        '''
	    byte GDOP_QC(TIME, DEPTH, LATITUDE, LONGITUDE) ;
        ncwrite(ncfile,'GDOP_QC',GDOP_QCflag');
        '''

        # Time quality flag:
        sdnTime_QCflag = 1
        self.variables['TIME_QC'] = xr.DataArray(sdnTime_QCflag, dims   = {'TIME' : 1}, coords = {'TIME' : self.variables['TIME']})
        
        # Position quality flag:
        sdnPosition_QCflag = totVel.copy()*0 + 1
        self.variables['POSITION_QC'] = sdnPosition_QCflag
      
        # Depth quality flag
        sdnDepth_QCflag = 1
        self.variables['DEPH_QC'] = xr.DataArray(sdnTime_QCflag, dims   = {'TIME' : 1}, coords = {'TIME' : self.variables['TIME']})

        # Velocity Threshold quality flags
        ## Velocity Threshold QC test
        err           = self.variables['err']

        velThr        = totVel.copy()
        velThr.values = np.select([(err == 0) | (err == 2), (err == 1) | (err == 3), np.isnan(err)], [1,4,np.nan])
        # velThr.values = np.select([totVel <= Total_QC_params.VelThr, totVel > Total_QC_params.VelThr, np.isnan(totVel)], [1,4,np.nan])

        '''
        % Velocity Threshold quality flags
        if (I(i) == 1 || I(i) == 3)
            velThr(lonGrid_idx,latGrid_idx,1) = 4;
        elseif (I(i) == 0 || I(i) == 2)
            velThr(lonGrid_idx,latGrid_idx,1) = 1;
        end
        '''
        self.variables['CSPD_QC'] = velThr

        # GDOP Threshold quality flags
        GDOPThr        = totVel.copy()
        GDOPThr.values = np.select([(err == 0) | (err == 1), (err == 2) | (err == 3), np.isnan(err)], [1,4,np.nan])

        '''
        if (I(i) == 2 || I(i) == 3)
            GDOPThr(lonGrid_idx,latGrid_idx,1) = 4;
        elseif (I(i) == 0 || I(i) == 1)
            GDOPThr(lonGrid_idx,latGrid_idx,1) = 1;
        end
        '''
        self.variables['GDOP_QC'] = GDOPThr

        # La variable err solo se usa en el control de calidad:
        var = self.variables.pop('err')

        # Data Density Threshold quality flag
        dataDens = totVel.copy()
        dataDens.values = np.select([DDENS >= Total_QC_params.DataDensityThr, DDENS < Total_QC_params.DataDensityThr, np.isnan(DDENS)], [1,4,np.nan]) 

        self.variables['DDNS_QC'] = dataDens

        # La varaible DDENS solo se usa en el control de calidad:
        var = self.variables.pop('DDENS')

        # Set the QC flag for the current hour to 0 (no QC performed)
        tempDer  = totVel.copy()
        tempDer *= 0

        self.variables['VART_QC'] = tempDer

        # Populate the overall quality variable:
        condicion = (self.variables['CSPD_QC'] == 1) & \
                    (self.variables['DDNS_QC'] == 1) & \
                    (self.variables['GDOP_QC'] == 1) & \
                    (self.variables['VART_QC'] == 1)

        isNan = np.isnan(self.variables['CSPD_QC'])

        self.variables['QCflag'] = self.variables['CSPD_QC'].copy()
        self.variables['QCflag'].values = np.select([condicion & ~isNan, ~condicion & ~isNan, isNan], [1,4,np.nan])

        # Terminamos ajustando algunos parámetros de las variables:
        for variable in ['TIME_QC', 'POSITION_QC', 'DEPH_QC', 'CSPD_QC', 'DDNS_QC', 'QCflag', 'GDOP_QC', 'VART_QC']:

                # Encoding de las variables en el fichero:
                self.variables[variable].encoding["scale_factor"] = scale_factors[variable]
                self.variables[variable].encoding["add_offset"  ] = add_offsets[variable]
                self.variables[variable].encoding["dtype"       ] = dtypes[variable]
                self.variables[variable].encoding["_FillValue"  ] = _FillValues[variable]

    def to_netcdf(self, fichero):

        radar = re.findall("[A-Z]{4}", fichero.split('/')[-1])[0]
        fecha = datetime.strptime('%s%s%s%s' % tuple(re.findall("\d+", fichero.split('/')[-1])),'%Y%m%d%H%M' ) 

        logging.info('Fichero: %s Radar: %s' % (fichero, radar))

        # Info de la proyección:
        self.variables['crs'] = xr.DataArray(np.int16(0),)

        # Datos SDN:
        ## Número de sites usados para obtener el total:
        numSites = len(self.tablas[1])

        SDN_EDMO_CODEs = {'PRIO' : 4841, 'SILL' : 2751, 'VILA' : 4841, 'LPRO' : 590, 'FIST': 2751 }
        codigos = np.array([SDN_EDMO_CODEs[site] for site in self.tablas[1].SITE],dtype=np.int16)
        codigos = np.expand_dims(codigos, axis=0)
        self.variables['SDN_EDMO_CODE'] = xr.DataArray(codigos, dims   = {'TIME' : 1, 'MAXINST' : numSites})

        cadena = b'HFR-Galicia'
        n      = len(cadena)
        self.variables['SDN_CRUISE'] = xr.DataArray(np.array([cadena]), dims   = {'TIME' : 1})
        
        # cadena = ('HFR-Galicia-%s' % radar).encode()
        cadena = b'HFR-Galicia-Total'
        n      = len(cadena)
        self.variables['SDN_STATION'] = xr.DataArray(np.array([cadena]), dims   = {'TIME' : 1})

        cadena = ('HFR-Galicia-Total_%sZ' % (self.TimeStamp.isoformat())).encode()
        n      = len(cadena)
        self.variables['SDN_LOCAL_CDI_ID'] = xr.DataArray(np.array([cadena]), dims   = {'TIME' : 1})

        cadena = b'http://opendap.intecmar.gal/thredds/catalog/data/nc/RADAR_HF/Galicia/catalog.html'
        n      = len(cadena)
        self.variables['SDN_REFERENCES'] = xr.DataArray(np.array([cadena]), dims   = {'TIME' : 1})

        cadena = b"<sdn_reference xlink:href=\"http://opendap.intecmar.gal/thredds/catalog/data/nc/RADAR_HF/Galicia/catalog.html\" xlink:role=\"\" xlink:type=\"URL\"/>"
        n      = len(cadena)
        self.variables['SDN_XLINK'] = xr.DataArray(np.array([[cadena]]), dims   = {'TIME' : 1, 'REFMAX' : 1})

        # Otras:
        # Por ahora forzamos a estos valores para probar pero hay que sacarlos de la segunda tabla:
        siteLats = np.empty(50)*np.nan
        siteLons = np.empty(50)*np.nan

        siteLats[0:numSites] = self.tablas[1].OLAT.values
        siteLons[0:numSites] = self.tablas[1].OLON.values

        siteLats = np.expand_dims(siteLats, axis=0)
        siteLons = np.expand_dims(siteLons, axis=0)

        # Las posiciones de emisión y recepción son las mismas:
        self.variables['SLTR'] = xr.DataArray(siteLats, dims   = {'TIME' : 1, 'MAXSITE' : 50})
        self.variables['SLNR'] = xr.DataArray(siteLons, dims   = {'TIME' : 1, 'MAXSITE' : 50})
        self.variables['SLTT'] = xr.DataArray(siteLats, dims   = {'TIME' : 1, 'MAXSITE' : 50})
        self.variables['SLNT'] = xr.DataArray(siteLons, dims   = {'TIME' : 1, 'MAXSITE' : 50})

        sites = np.array([site.ljust(15).encode() for site in self.tablas[1].SITE])
        sites = np.pad(sites,(0,50-len(sites)),'constant', constant_values=('', '               '))
        sites = np.expand_dims(sites,axis=0)

        self.variables['SCDR'] = xr.DataArray(sites, dims   = {'TIME' : 1, 'MAXSITE' : 50})
        self.variables['SCDT'] = xr.DataArray(sites, dims   = {'TIME' : 1, 'MAXSITE' : 50})

        self.variables['NARX'] = xr.DataArray([numSites], dims   = {'TIME' : 1})
        self.variables['NATX'] = xr.DataArray([numSites], dims   = {'TIME' : 1})

        for variable in ['SLTT','SLNT','SLTR','SLNR','NARX','NATX']:
                # Encoding de las variables en el fichero:
                self.variables[variable].encoding["scale_factor"] = scale_factors[variable]
                self.variables[variable].encoding["add_offset"  ] = add_offsets[variable]
                self.variables[variable].encoding["dtype"       ] = dtypes[variable]
                self.variables[variable].encoding["_FillValue"  ] = _FillValues[variable]

        # Generamos el xarra.Dataset. radial.variables contienen los xr.DataArray necesarios:
        dataset = xr.Dataset(self.variables)

        # Atributos globales:
        ## Leemos los atributos específicos de cada radar:
        f = open('%s.json' % radar)
        atributos = json.loads(f.read())
        f.close()

        ## Creamos algunos atributos:
        atributos['id'] = 'HFR-Galicia-Total_%sZ' % self.TimeStamp.isoformat()

        atributos['time_coverage_start'] = '%sZ' % (self.TimeStamp-timedelta(minutes=30)).isoformat()
        atributos['time_coverage_end']   = '%sZ' % (self.TimeStamp+timedelta(minutes=30)).isoformat()

        ahora = datetime(*datetime.now().timetuple()[0:6]).isoformat()
        atributos['date_created']        = '%sZ' % ahora
        atributos['metadata_date_stamp'] = '%sZ' % ahora
        atributos['date_modified']       = '%sZ' % ahora
        atributos['date_issued']         = '%sZ' % ahora

        atributos['history']             = '%s data collected. %s netCDF file created and sent to European HFR Node' % (self.TimeStamp.isoformat(), ahora)

        atributos['DoA_estimation_method'] = ';'.join([' %s: Direction Finding' % radar for radar in self.radares]).strip()

        correos = {'LPRO': 'margarida.alves@hidrografico.pt', 'PRIO': 'gis@intecmar.gal', 'FIST': 'maribel@puertos.es','SILL': 'maribel@puertos.es', 'VILA': 'gis@intecmar.gal'}
        atributos['calibration_link'] = ';'.join([' %s: %s' % (radar,correos[radar]) for radar in self.radares]).strip()

        atributos['calibration_type'] = ';'.join([' %s: APM' % radar for radar in self.radares]).strip()

        ## ... y los insertamos
        dataset.attrs = atributos

        # Atributos de las variables:
        f = open('variables.json')
        atributos = json.loads(f.read())
        f.close()

        # Los tipos de los atributos valid_min/max son deserializados incorrectamente:
        for var in dataset:
            for key, value in atributos[var].items():
                if isinstance(atributos[var][key], int):

                    # Generamos un conversor de tipo a partir del tipo de la variable:
                    conversor           = np.dtype(dtypes[var])
                
                    # Utilizamos el conversor para recodificar un tipo nativo de python a un escalar tipo numpy:
                    atributos[var][key] = np.int_(atributos[var][key]).astype(conversor)

                elif isinstance(atributos[var][key],list):
    
                    # Generamos un conversor de tipo a partir del tipo de la variable:
                    conversor           = np.dtype(dtypes[var])
                
                    # Utilizamos el conversor para recodificar un tipo nativo de python a un escalar tipo numpy:
                    atributos[var][key] = np.array(atributos[var][key]).astype(conversor)

        for var in dataset:
            dataset[var].attrs = atributos[var]

        # Completamos coordenadas y dimensiones que xArray procesa de forma automática una vez creado el xr.Dataset a partir del diccionario de variables:
        for var in ['TIME', 'DEPH']:

            dataset[var].encoding["dtype"       ] = dtypes[var]
            dataset[var].encoding["_FillValue"  ] = None

            dataset[var].attrs = atributos[var]

        for var in ['LONGITUDE','LATITUDE']:

            dataset[var].encoding["dtype"       ] = dtypes[var]
            dataset[var].encoding["_FillValue"  ] = _FillValues[var]

            dataset[var].attrs = atributos[var]

        for var in ['DEPH','LONGITUDE','LATITUDE']:

            # Generamos un conversor de tipo a partir del tipo de la variable:
            conversor           = np.dtype(dtypes[var])
        
            # Utilizamos el conversor para recodificar un tipo nativo de python a un escalar tipo numpy:
            dataset[var].attrs['valid_min'] = np.float_(atributos[var]['valid_min']).astype(conversor)
            dataset[var].attrs['valid_max'] = np.float_(atributos[var]['valid_max']).astype(conversor)
 
        # Escribimos el netCDF:
        dataset.reset_coords(drop=False).to_netcdf('HFR-Galicia-%s_%s.nc' % (radar, fecha.strftime('%Y_%m_%d_%H%M'))) 


    def __repr__(self):
    
        return '<Total class>'


class Total_QC_params():

    """
    Clase estática para contener los umbrales.
    """

    VelThr         = 1.2 # (m/s)
    tempDer_Thr    = 1   # Diferencia de 1 m/s
    GDOPThr        = 2
    DataDensityThr = 3

class Grid():

    """
    Clase para la generación de la malla para albergar los datos

    Atributos
    ---------
    longitud, latitud: Matriz con las longitudes y latitudes reconstruidas

    Metodos
    -------
    """

    def __init__(self, fichero):

        """
        Constructor

        Parametros
        ----------
        """

        '''
        nLON, nLAT =  47, 81

        self.nLON, self.nLAT = nLON, nLAT

        origen_lon, origen_lat = -11.327712, 40.354694

        longitud = np.arange(nLON)*0.0730114  + origen_lon 
        latitud  = np.arange(nLAT)*0.05401373 + origen_lat

        self.longitud = xr.DataArray(longitud, dims={'LONGITUDE' : nLON})
        self.latitud  = xr.DataArray(latitud , dims={'LATITUDE'  : nLAT})
        '''

        coordenadas = xr.open_dataset(fichero)

        nLON, = coordenadas.LONGITUDE.shape
        nLAT, = coordenadas.LATITUDE.shape

        self.nLON, self.nLAT = nLON, nLAT

        self.longitud = coordenadas.LONGITUDE
        self.latitud  = coordenadas.LATITUDE

    def __repr__(self):
    
        return '<Grid class -> nLON: %i, nLAT: %i>' % (self.nLON, self.nLAT)

def VART_QC(ficheros):

    datasets = [xr.open_dataset(fichero) for fichero in ficheros] 
    totales  = [np.sqrt(dataset.EWCT**2 + dataset.NSCT**2)[0,0].values for dataset in datasets]

    totVel2h, totVel1h, totVel = totales

    # Definición del umbral:
    tempDer_Thr = Total_QC_params.tempDer_Thr

    tempDer1h = np.full_like(totVel1h,4)

    condicion  = np.abs(totVel   - totVel1h) < tempDer_Thr
    condicion &= np.abs(totVel2h - totVel1h) < tempDer_Thr

    tempDer1h[condicion] = 1

    condicion = np.isnan(totVel) | np.isnan(totVel2h)

    tempDer1h[condicion] = 0 # Ojo!!! HFR_Node_tools lo tiene a 1 (dato correcto)!

    tempDer1h[np.isnan(totVel1h)] = np.nan

    datasets[1].VART_QC.values[0,0,:] = tempDer1h[:]

    # Redefinimos overall quality variable para incluir los cambios recientes en VART_QC:
    condicion = (datasets[1].variables['CSPD_QC'] == 1) & \
                (datasets[1].variables['DDNS_QC'] == 1) & \
                (datasets[1].variables['VART_QC'] == 1)
#                    (datasets[1].variables['GDOP_QC'] == 1) & \

    isNan = np.isnan(datasets[1].variables['CSPD_QC'])

    datasets[1].variables['QCflag'].values = np.select([condicion & ~isNan, ~condicion & ~isNan, isNan], [1,4,np.nan])

    datasets[1].to_netcdf('%s_new.nc' % ficheros[1].split('.')[0])

if __name__ == '__main__':

    file = r'TOTL_GALI_2021_12_23_0600.tuv'
    path_in = r'../../datos/radarhf_tmp/tuv'
    path_out = r'../../datos/radarhf_tmp/nc'
    fichero = path_in + '/' + file
    print(fichero)
    radar    = re.findall("[A-Z]{4}", fichero.split('/')[-1])[0]

    fecha    = datetime.strptime('%s%s%s%s' % tuple(re.findall("\d+", fichero.split('/')[-1])),'%Y%m%d%H%M' ) 
    print(radar, fecha,fichero)
    total = Total(fichero)

    # Metemos la tabla en la malla regular del fichero auxiliar:
    grd = Grid('coordenadas.nc')
    total.to_grid(grd)

    # Generamos las variables de control de calidad:
    total.QC_control()

    # Generamos el fichero NetCDF:
    total.to_netcdf(fichero)

    ficheros    = ['HFR-Galicia-%s_%s.nc' % (radar, (fecha + timedelta(hours=-i)).strftime('%Y_%m_%d_%H%M')) for i in range(3)]
    condiciones = [os.path.isfile(fichero) for fichero in ficheros]

    if np.all(condiciones):
        logging.info('Procesando VART_QC en %s' % ficheros[1])
        VART_QC(ficheros)
    else:
        logging.info('No VART_QC')

