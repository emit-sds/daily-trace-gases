import rasterio as rio
from rasterio.warp import calculate_default_transform
from georeader.readers import emit
import os
from typing import Tuple, Optional, Any, Union
import rasterio
import rasterio.windows
import numpy as np
from georeader.geotensor import GeoTensor
import rasterio.warp
from datetime import datetime
import netCDF4

def rio_load(path, verbose=False):
    with rio.open(path) as src:
        data = src.read()
        if verbose: print("crs", src.crs)
    return data

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def file_exists(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def folder_exists(folder_path):
    return os.path.isdir(folder_path)

def center_crop(data, w, h):
    if len(data.shape) == 3:  # assuming data in CH,W,H
        start_w = int((data.shape[1] - w) / 2)
        start_h = int((data.shape[2] - h) / 2)
        return data[:, start_w : start_w + w, start_h : start_h + h]
    elif len(data.shape) == 2:  # assuming data in W,H
        start_w = int((data.shape[0] - w) / 2)
        start_h = int((data.shape[1] - h) / 2)
        return data[start_w : start_w + w, start_h : start_h + h]
    else:
        assert False


def find_padding(v, divisor = 8):
    """
    Calculate the padding needed to make a given value divisible by a specified divisor.

    Args:
        v (int): The value to be padded.
        divisor (int, optional): The divisor to make the value divisible by. Default is 8.

    Returns:
        tuple: A tuple containing two integers, representing the padding to be added to the left and right (or top and bottom) respectively.
    """
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def pad_sample(tensor, divisor = 32, mode = "reflect"):
    """
    Predict on a tensor adding padding if necessary

    Args:
        tensor: np.array (C, H, W) of input values
        divisor: int
        mode: str ~ reflect or for example constant

    Returns:
        2D or 3D np.array with the prediction
    """
    assert len(tensor.shape) == 3, f"Expected 3D tensor, found {len(tensor.shape)}D tensor"

    pad_r = find_padding(tensor.shape[-2], divisor)
    pad_c = find_padding(tensor.shape[-1], divisor)

    # NOTE: Not sure if this is compatible between pytorch and numpy...
    tensor_padded = np.pad(tensor, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), mode)

    return tensor_padded

### GENERAL NC LOADER:
# Same as the default one, except it supports general layer name (e.g.: "mask" for L2A data)

class NCImage(emit.EMITImage):
    def __init__(self, filename: str, glt: Optional[GeoTensor] = None,
                 band_selection: Optional[Union[int, Tuple[int, ...], slice]] = slice(None),
                 layer_name = 'radiance', skip_wavelenghts=False):
        self.layer_name = layer_name
        ###
        self.filename = filename
        self.nc_ds = netCDF4.Dataset(self.filename, 'r', format='NETCDF4')
        self._nc_ds_obs: Optional[netCDF4.Dataset] = None
        self._nc_ds_l2amask: Optional[netCDF4.Dataset] = None
        self._observation_bands = None
        self._mask_bands = None
        self.nc_ds.set_auto_mask(False)  # disable automatic masking when reading data

        self._mean_sza = None
        self._mean_vza = None
        self.obs_file: Optional[str] = None
        self.l2amaskfile: Optional[str] = None

        self.real_transform = rasterio.Affine(self.nc_ds.geotransform[1], self.nc_ds.geotransform[2], self.nc_ds.geotransform[0],
                                              self.nc_ds.geotransform[4], self.nc_ds.geotransform[5], self.nc_ds.geotransform[3])

        self.time_coverage_start = datetime.strptime(self.nc_ds.time_coverage_start, "%Y-%m-%dT%H:%M:%S%z")
        self.time_coverage_end = datetime.strptime(self.nc_ds.time_coverage_end, "%Y-%m-%dT%H:%M:%S%z")

        self.dtype = self.nc_ds[self.layer_name].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds[self.layer_name]._FillValue
        self.nodata = self.nc_ds[self.layer_name]._FillValue
        self.units = self.nc_ds[self.layer_name].units

        if glt is None:
            glt_arr = np.zeros((2,) + self.nc_ds.groups['location']['glt_x'].shape, dtype=np.int32)
            glt_arr[0] = np.array(self.nc_ds.groups['location']['glt_x'])
            glt_arr[1] = np.array(self.nc_ds.groups['location']['glt_y'])
            # glt_arr -= 1 # account for 1-based indexing

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
            self.glt = GeoTensor(glt_arr, transform=self.real_transform,
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.spatial_ref),
                                 fill_value_default=0)
        else:
            self.glt = glt

        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        xmin, ymin, xmax, ymax = self._bounds_indexes_raw()  # values are 1-based!

        # glt has the absolute indexes of the netCDF object
        # glt_relative has the relative indexes
        self.glt_relative = self.glt.copy()
        self.glt_relative.values[0, self.valid_glt] -= xmin
        self.glt_relative.values[1, self.valid_glt] -= ymin

        self.window_raw = rasterio.windows.Window(col_off=xmin - 1, row_off=ymin - 1,
                                                  width=xmax - xmin + 1, height=ymax - ymin + 1)

        self.band_selection = band_selection

        if not skip_wavelenghts:
            if "wavelengths" in self.nc_ds['sensor_band_parameters'].variables:
                self.bandname_dimension = "wavelengths"
            elif "radiance_wl" in self.nc_ds['sensor_band_parameters'].variables:
                self.bandname_dimension = "radiance_wl"
            else:
                raise ValueError(f"Cannot find wavelength dimension in {list(self.nc_ds['sensor_band_parameters'].variables.keys())}")

            self.wavelengths = self.nc_ds['sensor_band_parameters'][self.bandname_dimension][self.band_selection]
            self.fwhm = self.nc_ds['sensor_band_parameters']['fwhm'][self.band_selection]
            self._observation_date_correction_factor: Optional[float] = None

    def load_raw(self, transpose: bool = True) -> np.array:
        slice_y, slice_x = self.window_raw.toslices()
        if isinstance(self.band_selection, slice):
            data = np.array(self.nc_ds[self.layer_name][slice_y, slice_x, self.band_selection])
        else:
            data = np.array(self.nc_ds[self.layer_name][slice_y, slice_x][..., self.band_selection])
        # transpose to (C, H, W)
        if transpose and (len(data.shape) == 3):
            data = np.transpose(data, axes=(2, 0, 1))
        return data

    def load(self, boundless:bool=True, as_reflectance:bool=False, orthorectify:bool=True)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        if as_reflectance:
            invalids = np.isnan(data) | (data == self.fill_value_default)
            from georeader import reflectance

            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(self.wavelengths, self.fwhm, thuiller["Nanometer"].values)
            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(response) / 1_000
            data = reflectance.radiance_to_reflectance(data, solar_irradiance_norm,
                                                       units=self.units,
                                                       observation_date_corr_factor=self.observation_date_correction_factor)
            data[invalids] = self.fill_value_default

        if orthorectify:
            return self.georreference(data, fill_value_default=self.fill_value_default)
        else:
            return GeoTensor(values=data, transform=self.transform, crs=self.crs,
                     fill_value_default=self.fill_value_default)

    def to_crs(self, crs: Any = "UTM",
               resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = 60) -> '__class__':
        import georeader
        if crs == "UTM":
            footprint = self.glt.footprint("EPSG:4326")
            crs = georeader.get_utm_epsg(footprint)

        glt = georeader.read.read_to_crs(self.glt, crs, resampling=rasterio.warp.Resampling.nearest,
                               resolution_dst_crs=resolution_dst_crs)

        out = NCImage(self.filename, glt=glt, band_selection=self.band_selection, layer_name=self.layer_name)
        # Copy _pol attribute if it exists
        if hasattr(self, '_pol'):
            setattr(out, '_pol', georeader.window_utils.polygon_to_crs(self._pol, self.crs, crs))

        return out

    def __copy__(self) -> '__class__':
        out = NCImage(self.filename, glt=self.glt.copy(), band_selection=self.band_selection, layer_name=self.layer_name)

        # copy nc_ds_obs if it exists
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out

    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        glt_window = self.glt.read_from_window(window, boundless=boundless)
        out = NCImage(self.filename, glt=glt_window, band_selection=self.band_selection, layer_name=self.layer_name)

        # copy attributes as in __copy__ method
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, self.nc_ds_obs)

        return out
