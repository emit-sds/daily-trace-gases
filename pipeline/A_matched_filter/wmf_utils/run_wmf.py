from pipeline.A_matched_filter.wmf_utils.wmf import WMF_computation
import georeader.read
from georeader.griddata import georreference
import rasterio
import numpy as np

from georeader.readers import emit
from georeader.readers.emit import EMITImage
from datetime import datetime, timezone
from georeader import reflectance
import georeader

def _load_RGB(input_netcdf):
    tile_name = input_netcdf.split("/")[-1].replace(".nc","")

    try:
        ei = emit.EMITImage(input_netcdf)
    except:
        print("failed in EMIT file", input_netcdf)
        assert False
    wavelengths_read = np.array([640, 550, 460])
    bands_read = np.argmin(np.abs(wavelengths_read[:, np.newaxis] - ei.wavelengths), axis=1).tolist()

    # Reproject to UTM
    crs_utm = georeader.get_utm_epsg(ei.footprint("EPSG:4326"))
    emit_image_utm = ei.to_crs(crs_utm)
    emit_image_utm_rgb = emit_image_utm.read_from_bands(bands_read)
    emit_image_utm_rgb_norefl = emit_image_utm_rgb.load(as_reflectance=False)
    d = emit_image_utm_rgb_norefl
    geo = emit_image_utm_rgb_norefl

    image_values = d.values
    fill_value_default = d.fill_value_default

    def convert_to_reflectance(image_values, fill_value_default, band_names, tile_date, center_coords):
        thuiller = reflectance.load_thuillier_irradiance()
        wavelengths = np.array([float(b.replace("nm", "")) for b in band_names])
        fwhm = np.array([8.46, 8.443, 8.426])  # < hardcoded for RGB

        srf = reflectance.srf(
            center_wavelengths=wavelengths,
            fwhm=fwhm,
            wavelengths=thuiller["Nanometer"].values,
        )  # (8191, K)

        solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(srf)  # mW/m$^2$/nm
        solar_irradiance_norm /= 1_000
        invalids = np.any(image_values == fill_value_default, axis=0)
        out_values = reflectance.radiance_to_reflectance(image_values, solar_irradiance_norm, tile_date,
                                                         center_coords=center_coords)

        fill_value_default = 0
        out_values = out_values * 10_000
        out_values[:, invalids] = fill_value_default

        return out_values

    band_names = ['640nm', '550nm', '460nm']
    tile_date = datetime.strptime(tile_name.split("_")[4], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    center_coords = geo.footprint("EPSG:4326").centroid.coords[0]  ### < this might potentially not be exact?
    out_values = convert_to_reflectance(image_values, fill_value_default, band_names, tile_date, center_coords)

    # Added check for nans:
    out_values = np.nan_to_num(out_values, nan=0)

    # Normalise
    RGB_NORMALIZATION_SATELLITE = 4_500
    normalised_rgb_values = np.clip(out_values.astype(np.float32) / RGB_NORMALIZATION_SATELLITE, 0, 1)
    # invalids are still at 0s

    return normalised_rgb_values, geo

def _compute_WMF(emit_image, toa_reflectances):
    wmf = WMF_computation(emit_image,
                           wavelengths=emit_image.wavelengths,
                           mean_vza=emit_image.mean_vza,
                           mean_sza=emit_image.mean_sza,
                           fwhm=emit_image.fwhm,
                           logger=None)

    wmf_geo = georreference(emit_image.glt_relative, wmf, emit_image.valid_glt,
                                     fill_value_default=emit_image.fill_value_default)

    # Correct to the same crs and transform
    wmf_geo = georeader.read.read_reproject_like(wmf_geo, toa_reflectances,
                                                 # resampling = rasterio.warp.Resampling.cubic_spline,
                                                 resampling=rasterio.warp.Resampling.nearest,
                                                 )
    wmf = wmf_geo.values

    # print("wmf", wmf.shape) # wmf (2239, 2554)
    wmf = np.expand_dims(wmf, axis=0) # wmf (1, 2239, 2554)

    # print("wmf values", np.min(wmf), np.mean(wmf), np.max(wmf))
    return wmf, wmf_geo

def save_product(save_path, data, ref_geo, descriptions=[], tags ={}, fill_value_default=0):
    from georeader.geotensor import GeoTensor
    from georeader.save import save_cog
    geo_data = GeoTensor(data, transform=ref_geo.transform, crs=ref_geo.crs, fill_value_default=fill_value_default)
    save_cog(geo_data, save_path, descriptions=descriptions, tags=tags)

def save_rgb_wmf_products(save_path_rgb, save_path_wmf, data_toa_radiances, data_wmf, wmf_geo_ref):
    # RGB
    if data_toa_radiances is not None:
        save_product(save_path_rgb, data_toa_radiances, wmf_geo_ref, descriptions=["red", "blue", "green"], tags={"units": "TOA_rad"}, fill_value_default=0)
    # WMF
    if data_wmf is not None:
        save_product(save_path_wmf, data_wmf, wmf_geo_ref, descriptions=["WMF"], tags={"units": "WMF"}, fill_value_default=-9999)

def run_wmf(save_path_wmf, save_path_rgb, input_netcdf):

    # REFLECTANCES DATA
    toa_reflectances, toa_geo_object = _load_RGB(input_netcdf)

    # COMPUTE WMF product
    emit_image = EMITImage(input_netcdf)
    wmf, wmf_geo = _compute_WMF(emit_image, toa_geo_object)

    save_rgb_wmf_products(save_path_rgb, save_path_wmf, toa_reflectances, wmf, wmf_geo)

