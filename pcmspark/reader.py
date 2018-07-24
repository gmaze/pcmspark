#!/usr/bin/env python
# encoding: utf-8
"""
readers.py

Class to read ISAS13, NATL12 and NATL60 dataset profiles in RDD

Created by Guillaume Maze on 2017-10-12.
Copyright (c) 2017 Guillaume Maze. All rights reserved.
"""
__author__ = 'gmaze@ifremer.fr'

import xarray as xr
from collections import OrderedDict

class IsasProfileReader:
    """Usage:
        Init the reader with subdomain parameters:
            reader = IsasProfileReader(dpt=slice(0,300),lat=slice(20,45),lon=slice(-180,-160))

        Create the RDD:
            rdd = sc.parallelize(flist).flatMap(reader('TEMP'))

        Access one profile with:
            reader('TEMP')(flist[0])[0,:]
    """

    def __init__(self, dpt=None, lat=None, lon=None):
        """Create the ISAS profile reader"""
        self.domain = {'dpt': dpt, 'lat': lat, 'lon': lon}

    def __call__(self, varname='TEMP'):
        """Return a function to be called by a rdd map or flatmap method"""

        def _processonefile(fpath):
            ds = xr.open_dataset(fpath)

            if varname in ds.data_vars:
                # Open and format the dataset:
                ds = self.dataset(fpath, varmask=varname)

                # Extract only the required variable:
                data = ds[varname].values.T
                print data.shape  # This will go into some log in: \
                # /home1/datawork/gmaze/spark-runtime/*.datarmor0/work/app-*/*/stderr

                # if sc.version == '2.1.0':
                #     from pyspark.mlib.linalg import Vectors
                # else:
                #     from pyspark.mllib.linalg import Vectors
                from pyspark.mllib.linalg import Vectors
                data = Vectors.dense(data)

            else:
                # Extract coordinates only:
                # ds = ds.isel(time=0)  # 1 time step per file
                ds = self._subselect(ds)
                data = ds[varname].values

            ds.close()
            return data

        return _processonefile

    def _subselect(self, ds):
        """Select a sub-domain of the dataset"""
        if self.domain['dpt']:
            ds = ds.sel(depth=self.domain['dpt'])
        if self.domain['lat']:
            ds = ds.sel(latitude=self.domain['lat'])
        if self.domain['lon']:
            ds = ds.sel(longitude=self.domain['lon'])
        return ds

    def shape(self,fpath,varname='TEMP'):
        """Return dimension as a dictionary"""
        ds = self.dataset(fpath,varmask=varname)
        D = {ds[dimension].name: ds[dimension].size for dimension in ('n_samples', 'depth')}
        return D

    def dataset(self, fpath, varmask='TEMP'):
        """Return data as xarray dataset
            It return the entire dataset but filtered through available points of varmask
        """

        # Open the file:
        ds = xr.open_dataset(fpath)

        # Select the sub-domain:
        # ds = ds.isel(time=0)  # 1 time step per file
        ds = self._subselect(ds)

        # Mask:
        mask = ds[varmask].notnull().sum(dim='depth') == len(ds['depth'])
        ds.coords['mask'] = (('time', 'latitude', 'longitude'), mask)

        # Stack the array along the sampling dimension:
        # To revert the operation: ds = ds.unstack(('n_samples'))
        ds = ds.stack(n_samples=('time', 'latitude', 'longitude'))

        # Select only full profiles
        # The "drop" option really reduces the number of samples (profiles) in the dataset
        ds = ds.where(ds.mask == 1, drop=True)

        return ds

    def new(self,fpath,varname='TEMP'):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varname)
        # Clean it up:
        for varname in ds.data_vars:
            ds = ds.drop(varname)
        ds.attrs = OrderedDict()
        return ds

    def add_feature(self, fpath, varname='TEMP', feature=None):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varname)

        # Add new feature:

        return ds

    def dataframe(self, rdd):
        """Transform the RDD into a dataframe"""
        from pyspark.sql import SparkSession
        from pyspark.sql import Row
        from pyspark.ml.linalg import Vectors

        data = rdd.map(lambda x: Row(temp=Vectors.dense(x)))
        dataFrame = spark.createDataFrame(data)

        return dataFrame

class Natl12ProfileReader:
    """Usage:
        Init the reader with subdomain parameters:
            reader = Natl12ProfileReader(dpt=slice(0,300),lat=slice(20,45),lon=slice(-180,-160))

        Create the RDD:
            rdd = sc.parallelize(flist).flatMap(reader('TEMP'))

        Access one profile with:
            reader('TEMP')(flist[0])[0,:]
    """

    def __init__(self, dpt=None, lat=None, lon=None):
        """Create the ISAS profile reader"""
        self.domain = {'dpt': dpt, 'lat': lat, 'lon': lon}

    def __call__(self, varname='votemper'):
        """Return a function to be called by a rdd map or flatmap method"""

        def _processonefile(fpath):
            ds = xr.open_dataset(fpath, decode_times=False)

            if varname in ds.data_vars:
                # Open and format the dataset:
                ds = self.dataset(fpath, varmask=varname)

                # Extract only the required variable:
                data = ds[varname].values.T
                print data.shape  # This will go into some log in: \
                # /home1/datawork/gmaze/spark-runtime/*.datarmor0/work/app-*/*/stderr

                from pyspark.mllib.linalg import Vectors
                data = Vectors.dense(data)

            else:
                # Extract coordinates only:
                # ds = ds.isel(time=0)  # 1 time step per file
                ds = self._subselect(ds)
                data = ds[varname].values

            ds.close()
            return data

        return _processonefile

    def _subselect(self, ds):
        """Select a sub-domain of the dataset"""
        if self.domain['dpt']:
            ds = ds.sel(deptht=self.domain['dpt'], drop=True)
        if self.domain['lat']:
            sl_y = self.domain['lat']
            if sl_y.step > 1:
                ds = ds.isel(y=slice(0, len(ds['y']), sl_y.step), drop=True)
            ds = ds.where((ds.nav_lat >= sl_y.start) & (ds.nav_lat <= sl_y.stop), drop=True)
        if self.domain['lon']:
            sl_x = self.domain['lon']
            if sl_x.step > 1:
                ds = ds.isel(x=slice(0, len(ds['x']), sl_x.step), drop=True)
            ds = ds.where((ds.nav_lon >= sl_x.start) & (ds.nav_lon <= sl_x.stop), drop=True)
        return ds

    def shape(self, fpath, varname='votemper'):
        """Return dimension as a dictionary"""
        ds = self.dataset(fpath, varmask=varname)
        D = {ds[dimension].name: ds[dimension].size for dimension in ('n_samples', 'deptht')}
        return D

    def dataset(self, fpath, varmask='votemper'):
        """Return data as xarray dataset
            It return the entire dataset but filtered through available points of varmask
        """

        # Open the file:
        ds = xr.open_dataset(fpath, decode_times=False)

        # Put lat/lon into the coordinates:
        ds.set_coords(('nav_lat', 'nav_lon'), inplace=True)

        # Select the sub-domain:
        # ds = ds.isel(time=0)  # 1 time step per file
        ds = self._subselect(ds)

        # Mask:
        mask = ds[varmask].notnull().sum(dim='deptht') == len(ds['deptht'])
        ds.coords['mask'] = (('time_counter', 'y', 'x'), mask)

        # Stack the array along the sampling dimension:
        # To revert the operation: ds = ds.unstack(('n_samples'))
        ds = ds.stack(n_samples=('time_counter', 'y', 'x'))

        # Select only full profiles
        # The "drop" option really reduces the number of samples (profiles) in the dataset
        ds = ds.where(ds.mask == 1, drop=True)

        return ds

    def new(self, fpath, varmask='votemper'):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varmask)
        # Clean it up:
        for varname in ds.data_vars:
            # if (varname not in ('nav_lon','nav_lat')):
            ds = ds.drop(varname)
        ds.attrs = OrderedDict()
        return ds

    def add_feature(self, fpath, varname='votemper', feature=None):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varname)

        # Add new feature:

        return ds

class Natl60ProfileReader:
    """Usage:
        Init the reader with subdomain parameters:
            reader = Natl60ProfileReader(dpt=slice(0,300),lat=slice(20,45),lon=slice(-180,-160))

        Create the RDD:
            rdd = sc.parallelize(flist).flatMap(reader('TEMP'))

        Access one profile with:
            reader('TEMP')(flist[0])[0,:]

        /home/datawork-lops-osi/data/natl60/NATL60-CJM165/1d/3D
    """

    def __init__(self, dpt=None, lat=None, lon=None):
        """Create the ISAS profile reader"""
        self.domain = {'dpt': dpt, 'lat': lat, 'lon': lon}
        self.mask_loaded = False

    def __call__(self, varname='votemper'):
        """Return a function to be called by a rdd map or flatmap method"""

        def _processonefile(fpath):
            ds = xr.open_dataset(fpath, decode_times=False)

            if varname in ds.data_vars:
                # Open and format the dataset:
                ds = self.dataset(fpath, varmask=varname)

                # Extract only the required variable:
                data = ds[varname].values.T
                print data.shape  # This will go into some log in: \
                # /home1/datawork/gmaze/spark-runtime/*.datarmor0/work/app-*/*/stderr

                from pyspark.mllib.linalg import Vectors
                data = Vectors.dense(data)

            else:
                # Extract coordinates only:
                # ds = ds.isel(time=0)  # 1 time step per file
                ds = self._subselect(ds)
                data = ds[varname].values

            ds.close()
            return data

        return _processonefile

    def _subselect(self, ds):
        """Select a sub-domain of the dataset"""
        if self.domain['dpt']:
            ds = ds.sel(deptht=self.domain['dpt'], drop=True)
        if self.domain['lat']:
            sl_y = self.domain['lat']
            if sl_y.step > 1:
                ds = ds.isel(y=slice(0,len(ds['y']),sl_y.step), drop=True)
            ds = ds.where((ds.nav_lat >= sl_y.start) & (ds.nav_lat <= sl_y.stop), drop=True)
        if self.domain['lon']:
            sl_x = self.domain['lon']
            if sl_x.step > 1:
                ds = ds.isel(x=slice(0,len(ds['x']),sl_x.step), drop=True)
            ds = ds.where((ds.nav_lon >= sl_x.start) & (ds.nav_lon <= sl_x.stop), drop=True)
        return ds

    def shape(self, fpath, varname='votemper'):
        """Return dimension as a dictionary"""
        ds = self.dataset(fpath, varmask=varname)
        D = {ds[dimension].name: ds[dimension].size for dimension in ('n_samples', 'deptht')}
        return D

    def dataset(self, fpath, varmask='votemper', vdrop=True):
        """Return data as xarray dataset
            It returns the entire dataset but filtered through available points of varmask
        """

        # Open the file:
        ds = xr.open_dataset(fpath, decode_times=False)

        # chunks = (1727, 2711) # cf https://github.com/pydata/xarray/issues/896
        # xr_chunks = {'x': chunks[-1], 'y': chunks[-2], 'time_counter': 1, 'deptht': 1}
        # ds = ds.chunk(xr_chunks)

        # Put lat/lon into the coordinates:
        ds.set_coords(('nav_lat', 'nav_lon'), inplace=True)

        # Remove everything but the necessary variables:
        if vdrop:
            for v in ds.data_vars:
                if not ((v in varmask) or (v in 'nav_lat') or (v in 'nav_lon') or (v in 'deptht')):
                    ds = ds.drop(v)

        # Select the sub-domain:
        # ds = ds.isel(time_counter=0)  # 1 time step per file
        ds = self._subselect(ds)

        # Mask:
        if not self.mask_loaded:
            mask = ds[varmask].notnull().sum(dim='deptht') == len(ds['deptht'])
            self.mask = mask
        else:
            mask = self.mask
        ds.coords['mask'] = (('time_counter', 'y', 'x'), mask)

        # Stack the array along the sampling dimension:
        # To revert the operation: ds = ds.unstack(('n_samples'))
        ds = ds.stack(n_samples=('time_counter', 'y', 'x'))

        # Select only full profiles
        # The "drop" option really reduces the number of samples (profiles) in the dataset
        #todo This is where everything is blown away with NATL60: masking is ok, but dropping raises an error
        ds = ds.where(ds.mask == 1, drop=True)
        # We could use a more binary and pragmatic approach here

        return ds

    def new(self, fpath, varmask='votemper'):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varmask)
        # Clean it up:
        for varname in ds.data_vars:
            # if (varname not in ('nav_lon','nav_lat')):
            ds = ds.drop(varname)
        ds.attrs = OrderedDict()
        return ds

    def add_feature(self, fpath, varname='votemper', feature=None):
        """Return empty dataset"""
        # Load the dataset:
        ds = self.dataset(fpath, varmask=varname)

        # Add new feature:

        return ds