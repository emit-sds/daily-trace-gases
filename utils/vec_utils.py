import numpy as np
import geopandas as gpd
from georeader.rasterio_reader import RasterioReader
from georeader import rasterize


def vec_load(vector_path):
    gdf = gpd.read_file(vector_path, use_arrow=True)
    return gdf

def how_many_pixels_does_polygon_occupy(polygon, reference_file="wmf.tif"):
    ref_data = RasterioReader(reference_file)
    rasterized = rasterize.rasterize_geometry_like(polygon, data_like=ref_data, value=1, fill=0, crs_geometry="EPSG:4326")
    number_of_pixels = np.sum(rasterized.values)
    return number_of_pixels, rasterized.values
