# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497

"""
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
#from scipy.spatial import distance
from math import sqrt

"""
RUTA: ¡CAMBIAR!
"""
workpath = ""
# "C:/Users/daniel/Desktop/CARRERA/4º/Geometría Computacional/Practica 4 - PCA y Analogia"
os.getcwd()
#os.chdir(workpath)
files = os.listdir(workpath)


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file(workpath + "/hgt.2019.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
level_units = f.variables['level'].units # 1mb = 1hPa
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
hgt_scale = f.variables['hgt'].scale_factor
hgt_offset = f.variables['hgt'].add_offset
print(hgt.shape)

f.close()

"""
Ejemplo de evolución temporal de un elemento de aire
"""
plt.plot(time, hgt_offset + hgt[:, 5, 0, 0]*hgt_scale, c='r')
plt.show()


dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) 
           for t in time]

"""
Distribución espacial de la altura geopotencial en el nivel de 1000hPa, para
el día 0.
"""
#import geopandas
#world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#lons = np.array([l if l<=180 else l-360 for l in lons])
#lons = np.sort(lons)

fig, ax = plt.subplots()
#csf = ax.contourf(lons, lats, hgt[0,0,:,:], 15, cmap='jet',alpha=1)
#world.plot(facecolor="none", edgecolor="black",ax=ax, alpha=.5)
cs = ax.contour(lons, lats, hgt[0,0,:,:])
#cs.levels = (cs.levels * hgt_scale) + hgt_offset
#ax.clabel(cs, cs.levels, fmt='%0.fmgp')
plt.show()

"""
Sistema de 365 elementos días y 10512 variables de estado.
"""
hgt2 = hgt[:,5,:,:].reshape(len(time),len(lats)*len(lons))

"""
Aplicación de PCA/EOF:
1. Sobre el sistema formado por 10512 elementos de aire con 365 variables de estado.
2. Sobre el sistema con 365 elementos días con 10512 variables de estado.
"""
n_components = 4

X = hgt2
Y = hgt2.transpose()
pca = PCA(n_components=n_components)


# Aplicamos PCA sobre matriz de elementos días.
pca.fit(X)
prop_explained_variance = pca.explained_variance_ratio_
print('Resumen temporal en vectores espaciales'.upper())
print('Proporción varianza explicada por componentes: ', prop_explained_variance)
print('Varianza explicada total: ', sum(prop_explained_variance))
acumulate_variance = [sum(prop_explained_variance[:j+1])
                            for j in range (len(prop_explained_variance))]
out = pca.singular_values_


fig = plt.figure()
plt.plot(np.arange(1,n_components+1), acumulate_variance, ls='-',color='k',
         marker='o', mec='k', mfc='w')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza explicada')
plt.xticks(np.arange(1,n_components+1))
plt.grid()
plt.show()


State_pca = pca.fit_transform(X).transpose()

# Representación temporal de las 4 componentes.
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    plt.plot(time, State_pca[i-1], c='r')
plt.suptitle('Representación temporal componentes')
plt.show()


# Aplicamos PCA sobre la matriz de elementos de aire.
pca.fit(Y)
prop_explained_variance = pca.explained_variance_ratio_
print('Resumen espacial en vectores temporales'.upper())
print('Proporción varianza explicada por componentes: ', prop_explained_variance)
print('Varianza explicada total: ', sum(prop_explained_variance))
acumulate_variance_ratio = [sum(prop_explained_variance[:j+1])
                            for j in range (len(prop_explained_variance))]
out = pca.singular_values_
A = pca.components_


fig = plt.figure()
plt.plot(np.arange(1,n_components+1), acumulate_variance_ratio, ls='-',color='k',
         marker='o', mec='k', mfc='w')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza explicada')
plt.xticks(np.arange(1,n_components+1))
plt.grid()
plt.show()


Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

# Representación espacial de las 4 componentes.
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()


"""
Búsqueda del los 4 días más análogos.
"""
# Extraemos los datos de Enero de 2020.
f = nc.netcdf_file(workpath + "/hgt.2020.nc", 'r')

time_20 = f.variables['time'][:].copy()
time_bnds_20 = f.variables['time_bnds'][:].copy()
time_units_20 = f.variables['time'].units
level_20 = f.variables['level'][:].copy()
level_units_20 = f.variables['level'].units # 1mb = 1hPa
lats_20 = f.variables['lat'][:].copy()
lons_20 = f.variables['lon'][:].copy()
hgt_20 = f.variables['hgt'][:].copy()
hgt_units_20 = f.variables['hgt'].units
hgt_scale_20 = f.variables['hgt'].scale_factor
hgt_offset_20 = f.variables['hgt'].add_offset

f.close()

# Fechas de 2020, longitudes y latitudes del subsistema.
dt_time_20 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
              for t in time_20]
lons = np.array([l if l<=180 else l-360 for l in lons]) # Cambio en longitudes.
lats_mask = (lats>30) & (lats<50)
lons_mask = (lons>-20) & (lons<20)

# Extracción del día de 2020 requerido, delimitando sus coordenadas al subsistema.
index = dt_time_20.index(dt.date(2020,1,20))
day_1000hPa = hgt_20[index,0,lats_mask,:][:,lons_mask].reshape(7*15).astype(np.int64)
day_500hPa = hgt_20[index,5,lats_mask,:][:,lons_mask].reshape(7*15).astype(np.int64)
#day = np.array([day_1000hPa, day_500hPa]).transpose().astype(np.int64)

# Extracción del subsistema de 2019.
hgt_subset = (hgt[:,:,lats_mask,:][:,:,:,lons_mask].reshape(365,17,7*15).astype(np.int64))
hgt_subset_1000hPa = hgt_subset[:,0,:]
hgt_subset_500hPa = hgt_subset[:,5,:]

# Cálculo de las distancias y posterior selección de los 4 días más análogos.
distances = []
for day in range (len(hgt_subset)):
    aux1 = 0.5 * (day_1000hPa - hgt_subset_1000hPa[day]) ** 2
    aux2 = 0.5 * (day_500hPa - hgt_subset_500hPa[day]) ** 2
    distances.append(sqrt(sum(aux1 + aux2)))
    
day_dist = list(zip(dt_time, distances))
day_dist.sort(reverse=False, key = lambda tup: tup[1])
days_analogs = [i for i, j in day_dist[:4]]

print('Días más análogos'.upper())
for i in range(4): print(days_analogs[i])

"""
Comparación gráfica de a_0 con su día más similar de 2019.
"""
# Reorganizaión de longitudes y alturas geopotenciales para representar después.
l = np.append(lons[lons_mask][8:], lons[lons_mask][:8])
Z1 = [np.append(lat[8:], lat[:8]) for lat in day_1000hPa.reshape(7,15)]
Z2 = [np.append(lat[8:], lat[:8]) for lat in 
      hgt_subset_1000hPa[dt_time.index(days_analogs[0])].reshape(7,15)]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5,5))
# Día más análogo.
ax1.contourf(l, lats[lats_mask], Z1,10,cmap='hot')
ax1.contour(l, lats[lats_mask], Z1,10,linestyles='solid', 
            colors='k',linewidths=.5)
ax1.set_title('2020-01-20')
# Día a_0
ax2.contourf(l, lats[lats_mask], Z2, 10,cmap='hot')
ax2.contour(l, lats[lats_mask], Z2,10,linestyles='solid', 
            colors='k',linewidths=.5)
ax2.set_title(str(days_analogs[0]))
fig.suptitle('Comparación con día más similar', fontweight='bold')

plt.show()

"""
Cálculo del error absoluto medio de la temperatura prevista.
"""
f = nc.netcdf_file(workpath + "/air.2019.nc", 'r')

air = f.variables['air'][:].copy()
air_units = f.variables['air'].units
air_scale = f.variables['air'].scale_factor
air_offset = f.variables['air'].add_offset

f.close()

air = air.astype(np.int32)
air = air * air_scale + air_offset

# Obtenemos la temperatura media de los 4 días en cada coordenada.

index_mask = [dt_time.index(day) for day in days_analogs]
air_analogs_1000hPa = air[index_mask,0,:,:][:,lats_mask,:][:,:,lons_mask].astype(np.int64)
air_analogs_500hPa = air[index_mask,5,:,:][:,lats_mask,:][:,:,lons_mask].astype(np.int64)
    
air_means = (0.5 * sum(air_analogs_1000hPa) + 0.5 * sum(air_analogs_500hPa)) / 4

# Extraemos datos de temperatura para el día a_0 de 2020.

f = nc.netcdf_file(workpath + "/air.2020.nc", 'r')

time_20 = f.variables['time'][:].copy()
time_bnds_20 = f.variables['time_bnds'][:].copy()
time_units_20 = f.variables['time'].units
air_20 = f.variables['air'][:].copy()
air_units_20 = f.variables['air'].units
air_scale_20 = f.variables['air'].scale_factor
air_offset_20 = f.variables['air'].add_offset

f.close()

dt_time_20 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
              for t in time_20]
air_20 = air_20.astype(np.int32)
air_20 = air_20 * air_scale_20 + air_offset_20

# Obtenemos subsistema.
air_day_1000hPa = air_20[index,0,lats_mask,:][:,lons_mask].astype(np.int64)
air_day_500hPa = air_20[index,5,lats_mask,:][:,lons_mask].astype(np.int64)
air_day = 0.5 * air_day_1000hPa + 0.5 * air_day_500hPa

# Cálculo de error.
error = abs(air_day - air_means)
print('Error absoluto medio: '.upper())
print(error)
print(error.mean())