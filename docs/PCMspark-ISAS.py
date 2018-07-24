# coding: utf-8
__author__ = 'gmaze@ifremer.fr'

#
#  Profile Classification Modelling with pyspark
#
# Demonstrate the workflow with ISAS data

from pyspark import SparkContext, SparkConf

# "sc" is the SparkContext
sc = SparkContext.getOrCreate()
print sc.master
print sc.version
sc._conf.getAll()
sc.setLogLevel('ERROR')

# Import python stuff
import sys
import os
import glob
import pandas as pd
import xarray as xr
import dask
import numpy as np
import matplotlib
matplotlib.use('agg')
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# # Load my machinery
# this avoid having too long notebooks

# In[219]:

# This should be done with a proper import but I'd rather execute the files so that
# I can easily reload the package if edited elsewhere
# os.chdir('/home1/datahome/gmaze/work/Projects/Oceans_Big_Data_Mining/datarmor/jynb_work')
os.chdir('/home1/datawork/gmaze/Projects/Oceans_Big_Data_Mining/pyspark/jynb_work')
execfile('pcmspark/common.py')
execfile('pcmspark/reader.py')

# In[Define the dataset to work with]:

# this is where we select the files to be used in the dataset
# it allows to pre-determine the size the dataset
dsname = 'ISAS13'

data_root = '/home1/datawork/gmaze/spark/ISAS/field/' # from Datarmor
# data_root = '/home/jovyan/data/ISAS/ANA_ISAS13/field/' # from my laptop/docker

# dsname = dsname + ', 1 year, 1 month'
# flist = glob.glob(data_root + '2012/ISAS13_20120615_fld_TEMP.nc')  # 1 year, 1 month

# dsname = dsname + ', 1 year, 12 months'
# flist = glob.glob(data_root + '2012/ISAS13_2012*15_fld_TEMP.nc')  # 1 year, all months

# dsname = dsname + ', all years, 1 month'
# flist = glob.glob(data_root + '*/ISAS13_20*0115_fld_TEMP.nc')  # all years, 1 month

dsname = dsname + ', all years, 12 months'
flist = glob.glob(data_root + '*/ISAS13_20*15_fld_TEMP.nc')  # all years, all months

# In[Create RDD pointing to temperature data]:

# {'depth': 102, 'n_samples_per_file': 2 958}
reader_small = IsasProfileReader(dpt=slice(0,1000),lat=slice(20,45),lon=slice(-80,-40))

# {'depth': 152, 'n_samples_per_file': 20 987}
# reader_medium = IsasProfileReader(lat=slice(20,45))
reader_medium = IsasProfileReader(dpt=slice(0,1000),lat=slice(0,60),lon=slice(-80,0))

# {'depth': 152, 'n_samples_per_file': 199 657}
reader_large = IsasProfileReader()

# Select the reader to use:
reader = reader_small
reader = reader_large
# reader = IsasProfileReader(dpt=slice(0,1000))

# Size:
shape = reader.shape(flist[0])
shape['n_samples_per_file'] = shape['n_samples']
shape['n_samples'] = shape['n_samples']*len(flist)
print "Will load a dataset of size:\n\t", shape

rdd_data = sc.parallelize(flist).flatMap(reader('TEMP'))
first = rdd_data.first()

# In[Scaling]:

# Compute scaling parameters:
from pyspark.mllib.feature import StandardScaler, StandardScalerModel

scaler = StandardScaler(withMean=True, withStd=True).fit(rdd_data)

sample_mean = scaler.call('mean')

# Effectively scale the dataset:
rdd_norm = scaler.transform(rdd_data)

# In[Reduction]:

# Compute PCA new dimensions:
from pyspark.mllib.feature import PCA as PCAmllib

Neof = 20
reducer = PCAmllib(Neof).fit(rdd_norm)
# print type(reducer)

# Effectively reduce the dataset:
rdd_reduced = reducer.transform(rdd_norm)
# print type(rdd_reduced)


# In[Classification with k-mean]:

### Lancement de KMean pour creation du modele de classification
from pyspark.mllib.clustering import KMeans as KMeansmllib
import time
start_time = time.time()

NBCLUSTERS=8
INITMODE='kmean||'   # kmean|| or random
clusters_kmean = KMeansmllib.train(
                rdd_reduced,
                NBCLUSTERS, maxIterations=200, runs=20,
                initializationMode=INITMODE)

elapsed_time = time.time() - start_time
# print "Time to classify with kmean: ", elapsed_time

# Time to classify with kmean:  1564.55499506
# >>> 1564.55499506/60
# 26.075916584333335
# >>> shape
# {'deptht': 35, 'n_samples_per_file': 854,108, 'n_samples': 204,985,920}

# Classify
res_classif_kmean = clusters_kmean.predict(rdd_reduced)

# In[Plot results]:

# Simply plot the first map of labels:

# Get empty dataset with correct coordinates:
# ds_src = reader.new(flist[0],varmask='votemper')
ds_src = reader.dataset(flist[0], varmask='TEMP')
n_samples = ds_src['n_samples'].values.shape[0]
# Create dataset with first map of labels:
ic = 0
LABELS = res_classif_kmean.take((1+ic)*n_samples)
ds_lab = xr.Dataset({'label': (['n_samples'], LABELS)},
            coords={'n_samples': ds_src['n_samples']}).unstack(('n_samples'))
ds = xr.merge([ds_src.unstack(('n_samples')), ds_lab])
# ds['longitude'] = ds['longitude'].squeeze(dim='time', drop=True)

x = ds['longitude'].values
y = ds['latitude'].values
c = ds['label'].isel(time=0).values

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90,0,0,70], ccrs.PlateCarree())
ax.contourf(x, y, c, transform=ccrs.PlateCarree(), levels=np.arange(0, NBCLUSTERS))
# ax.coastlines()
plt.savefig('labelsISAS13.pdf', format='pdf')
# plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format=None,
#         transparent=False, bbox_inches=None, pad_inches=0.1,
#         frameon=None)

# Save labels:
ds_lab.to_netcdf('ISAS13.nc')

# RECAP:
sc._conf.getAll()
print "1st Profile data:\n", first # First profile in the collection
print "Sample mean profile: ", sample_mean
print ("Working dataset:\n\t- %s\n\t- %i files\n\t- %s")%(dsname,len(flist),bytes_2_human_readable(flist_size(flist)))
print "Will load a dataset of size:\n\t", shape
print "Time to classify with kmean: ", elapsed_time
print ds
print x.shape, y.shape, c.shape


# Perf
# ====
# Working dataset:
# 	- ISAS13, all years, 12 months
# 	- 132 files
# 	- 34.2 GB

# qsub -I -q mpi_8 -l walltime=4:0:0
# {'depth': 102, 'n_samples_per_file': 215173, 'n_samples': 28402836}
# Time to classify with kmean:  245.749878883