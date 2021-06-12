"""
dlav.py
Defines methods for loading and visualizing geospatial data.
Author: Ngoc Hoang
Last modified: June 9, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from sklearn.preprocessing import KBinsDiscretizer
import math

class FileException(Exception):
    """Exception to handle file problems."""
    pass


def dlav_geotiff(filenames):
    """Loads and displays information of .tiff files.

    Args:
        filenames: list of names of files to be loaded

    Returns:
        [PLACEHOLDER]
    """
    for file in filenames:
        print(f"Inspecting file: {file}\n")
        # Return GDALDataset object
        dataset = gdal.Open(file, gdal.GA_ReadOnly)
        # Raise exception for file not found error
        if not dataset:
            raise FileException(f"File {file} not found. Please check again.")

        # Display some general information
        print("General information:")
        display_info_geotiff(dataset)
        print('\n')

        # Display some raster bands information
        print("Raster bands information:")
        display_raster_bands_info(dataset)
        print('\n')

        # Visualize raster bands
        print("Raster bands visualizations:")
        visualize_raster_bands(dataset)
        print('\n')

def display_info_geotiff(dataset):
    """Displays information of a loaded raster dataset information.
    Reference: https://gdal.org/tutorials/raster_api_tut.html - Getting Dataset Information

    Args:
        dataset: GDALDataset object

    Returns:
        [PLACEHOLDER]
    """
    print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName))

    print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
    print("Projection is {}".format(dataset.GetProjection()))
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

def display_raster_bands_info(dataset):
    """
    Iterates through all raster bands of a dataset
    to fetch GDALRasterBand object and displays some information.
    Raster bands are numbered starting at 1.
    Reference: https://gdal.org/tutorials/raster_api_tut.html - Fetching a Raster Band

    Args: 
        dataset: GDALDataset object
    
    Returns:
        [PLACEHOLDER]
    """
    raster_count = dataset.RasterCount
    # Iterate from band #1 to band #raster_count
    for i in range(1, raster_count+1):
        print(f"Band #{i}:")
        band = dataset.GetRasterBand(i)
        print(f"Band type = {gdal.GetDataTypeName(band.DataType)}")
        
        min = band.GetMinimum()
        max = band.GetMaximum()
        if not min or not max:
            (min, max) = band.ComputeRasterMinMax(True)
        print(f"Min = {min:.3f}, max = {max:.3f}")

        if band.GetOverviewCount() > 0:
            print(f"Band has {band.GetOverviewCount()} overviews.")

        if band.GetRasterColorTable():
            print(f"Band has a color table with {band.GetRasterColorTable().GetCount()} entries.")

def visualize_raster_bands(dataset):
    """
    Iterates through all raster bands of a dataset
    and makes plots.

    Args:
        dataset: GDALDataset object
    
    Returns:
        [PLACEHOLDER]
    """
    raster_count = dataset.RasterCount
    # Iterate from band #1 to band #raster_count
    bands = []
    for i in range(1, raster_count+1):
        band = dataset.GetRasterBand(i)
        bands.append(band)
    arrs = [band.ReadAsArray() for band in bands]
    fig, axs = plt.subplots(1, raster_count, figsize=(8*raster_count, 8))
    for i in range(raster_count):
        if raster_count > 1:
            axs[i].imshow(arrs[i], cmap='GnBu')
            axs[i].set_title(f"Band #{i+1}")
        else:
            plt.imshow(arrs[0])
            plt.title(f"Band #1")
    plt.show()

def dlav_shapefile(filenames, n_discrete=10, exclude=[]):
    """Loads .shp files.

    Args:
        filenames: list of names of files to be loaded
        n_discrete: threshold for the number of unique values of an attribute
                    for discretizing the values into classes
        exclude: list of attributes to exclude, none by default

    Returns:
        [PLACEHOLDER]
    """
    for file in filenames:
        print(f"Inspecting file: {file}\n")
        # Return geopandas.GeoDataFrame object
        # File not found error is already handled by geopandas
        dataframe = gpd.read_file(file)

        # Inspect file attributes
        attrs = list(dataframe.columns)
        if 'geometry' in attrs:
            attrs.remove('geometry')
        for col in exclude:
            if col in attrs:
                attrs.remove(col)
        attrs_with_stat = list(dataframe.describe().columns)
        attrs_without_stat = [attr for attr in attrs if attr not in attrs_with_stat]
        print(f"List of attributes: {attrs}")
        # Attributes with statistical information
        for attr in attrs:
            print(f"Inspecting attribute: {attr}")
            print(f"Attribute values:")
            col = dataframe[attr]
            if attr in attrs_with_stat:
                describe_attr(col.describe(), stat=True)
            elif attr in attrs_without_stat:
                print(display_value_counts(col))
                describe_attr(col.describe(), stat=False)
            bin_counts = len(col.value_counts())
            if bin_counts <= n_discrete:
                visualize_attr(col)
                print(f"Attribute plot:")
                plot_attr(dataframe, attr)
            if bin_counts > n_discrete and attr in attrs_with_stat:
                print(f"Number of values exceeds threshold ({n_discrete}), discretizing:")
                new_classes, title_value = discretize_attr(col)
                if bin_counts < 500:
                    visualize_discrete_attr(col, title_value)
                print(f"Attribute plots (before and after discretization):")
                plot_discrete(dataframe, attr, new_classes)

def describe_attr(col, stat):
    """Displays information and visualizes a column in a dataset

    Args:
        col: PandasSeries object obtained from DataFrame.describe()
            attrs: count, mean, std, min, max, 25%, 50%, 75%
        stat: boolean value as to whether the attribute has statistical info

    Returns:
        [PLACEHOLDER]
    """
    if stat: # For attributes with statistical information
        print(f"Value count: {int(col.loc['count'])}")
        print(f"Mean value: {col.loc['mean']:.3f}")
        print(f"Standard deviation: {col.loc['std']:.3f}")
        print(f"Min: {col.loc['min']}")
        print(f"Max: {col.loc['max']}")
        print(f"Percentile: 25%: {col.loc['25%']}; 50%: {col.loc['50%']}; 75%: {col.loc['75%']}\n")
    else: # For attributes without statistical information
        print(f"Value count: {int(col.loc['count'])}")
        print(f"Number of unique values: {int(col.loc['unique'])}")
        print(f"Most frequent value: {col.loc['top']} (freq. {col.loc['freq']})\n")        

def display_value_counts(col):
    """Displays the unique values and frequencies in a particular column

    Args:
        col: PandasSeries object that is a column in a dataset
    
    Returns:
        count_df: PandasDataFrame object containing unique values in the column
                  and their frequencies sorted by frequencies in descending order
    """
    counts = col.value_counts()
    counts.sort_values(ascending=False, inplace=True)
    df = {'Value': counts.index, 'Frequency': counts.values}
    count_df = pd.DataFrame(df)
    return count_df

def visualize_attr(col):
    """Plots a column by value counts

    Args:
        col: PandasSeries object that is a column in a dataset

    Returns:
        [PLACEHOLDER]
    """
    # NaN values are currently dropped
    value_counts = col.value_counts()
    value_counts.sort_index(inplace=True)
    fig = plt.figure(figsize=(16, 8))
    # num_ticks = len(value_counts)
    # plt.locator_params(axis="x", nbins=num_ticks)
    plt.bar(value_counts.index, value_counts.values)

    plt.title(f"Graph of values & their frequencies in the {col.name} column")
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.show()

def discretize_attr(col, n_bins=1):
    """Discretizes a column with a specified number of classes/bins

    Args:
        col: PandasSeries object that is a column in a dataset
        n_bins: number of classes to discretize the values into,
                n_bins >= 2
    
    Returns:
        new_classes: PandasSeries object of the class identifier
                     encoded with integer values
        title_value: PandasDataFrame object for visualization
    """
    value_counts = col.value_counts()
    value_counts.sort_index(inplace=True)

    if n_bins == 1:
        # Default n_bins value of 1 = no value specified
        # Currently: take bin width to be standard deviation
        # Bin number = (max - min) / width
        std, max, min = col.describe()[['std', 'max', 'min']]
        n_bins = math.ceil((max - min) / std)

    print(f"Discretizing into {n_bins} bins.")

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    identifier = discretizer.fit_transform(value_counts.index.values.reshape(-1, 1))
    clf_df = pd.DataFrame(zip(value_counts.index, value_counts.values, identifier.flatten()), 
                              columns=['index', 'value', 'class'])
    gr = clf_df.groupby('class')
    agg = gr.agg({'index': np.mean, 'value': sum})
    agg['first'] = gr.first()['index']
    agg['last'] = gr.last()['index']
    agg['title'] = agg['first'].astype(int).astype(str) + ' - ' + agg['last'].astype(int).astype(str)

    id_dict = clf_df[['index', 'class']].set_index('index').to_dict()
    new_classes = col.map(id_dict['class'])
    title_value = agg[['title', 'value']].set_index('title')
    
    return new_classes, title_value

def visualize_discrete_attr(col, title_value):
    """

    Args:
        col: PandasSeries object that is a column in the dataset
        title_value: PandasDataFrame object obtained from discretizing col

    Returns:
        [PLACEHOLDER]
    """
    value_counts = col.value_counts()
    value_counts.sort_index(inplace=True)

    fig, axs = plt.subplots(2, 1, figsize=(16, 18))
    # plt.xticks(ticks=[])
    axs[0].locator_params(axis="x", nbins=len(value_counts))
    axs[0].bar(value_counts.index, value_counts.values)
    axs[0].set_title(f"Graph of values & their frequencies in the {col.name} column (before discretization)")
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].locator_params(axis="x", nbins=len(title_value))
    axs[1].set_title(f"Graph of values & their frequencies in the {col.name} column (after discretization)")
    axs[1].bar(title_value.index, title_value.value)
    axs[1].set_xlabel('Value')

    plt.show()

def plot_attr(df, col):
    """Plots a specific attribute of a dataframe

    Args:
        df: GeoPandasDataFrame object containing the dataset
        col: name of the specific column to be plotted (String)
    
    Returns:
        [PLACEHOLDER]
    """
    f, ax = plt.subplots(1, figsize=(8, 8))
    ax = df.plot(ax=ax, column=col, legend=True, legend_kwds={'label': col})
    plt.title(f"BY {col.upper()}")
    plt.show()

def plot_discrete(df, col, new_classes):
    """
    Plots a specific attribute of a dataframe
    that has been discretized (shows before and after)

    Args:
        df: GeoPandasDataFrame object containing the dataset
        col: name of the specific column to be plotted (String)
        new_classes: PandasSeries object of the class identifier
                     encoded with integer values
    
    Returns:
        [PLACEHOLDER]
    """
    new_df = df.copy(deep=True)
    new_df['class'] = new_classes

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    new_df.plot(ax=axs[0], column=col, legend=True, legend_kwds={'label': col})
    axs[0].set_title(f"BY {col.upper()} (BEFORE DISCRETIZATION)")
    new_df.plot(ax=axs[1], column='class', legend=True, legend_kwds={'label': 'Class'})
    axs[1].set_title(f"BY {col.upper()} (AFTER DISCRETIZATION)")
    plt.show()
    # =====================
    # Note: to change the legend labels to proper agg classes (using dict)
    # =====================

def test():
    # df = gpd.read_file('Conwy.shp')
    # lu = df['LU']
    # lu_info = df.describe()['LU']
    # describe_attr(lu_info)
    # visualize_attr(lu)
    # load_shapefile(['Conwy.shp'])
    dlav_shapefile(['Conwy.shp'])