# SOURCE > https://github.com/UNEP-IMEO-MARS/marsml-hyperspectral/blob/main/data/matched_filter_utils/wmf.py
# This module is a modified version of the Wide Matched Filter (WMF) retrieval method of Roger et al. 2024
# Roger, J., Guanter, L., Gorroño, J., and Irakulis-Loitxate, I.:
# Exploiting the entire near-infrared spectral range to improve the detection of methane plumes with high-resolution imaging spectrometers,
# Atmos. Meas. Tech., 17, 1333–1346, https://doi.org/10.5194/amt-17-1333-2024, 2024.

import os
import numpy as np
from georeader.readers import emit
from typing import Tuple, Optional
import logging
from numpy.typing import NDArray
from pipeline.A_matched_filter.wmf_utils.wmf_utils import calc_jac_rad, generate_filter, read_luts, AT_MF_select_window_alt

FILE_LUT_GAS = os.path.join(os.path.dirname(__file__), "output_Tch4_LUT_AMF_VZA_0_v2.nc")
def load_target_spectrum_mf(wavelengths, mean_vza, mean_sza, fwhm)-> NDArray:

    amf = 1. / np.cos(mean_vza * np.pi /180) + 1. / np.cos(mean_sza * np.pi /180)
    wvl_mod, t_gas_arr, mr_gas_arr = read_luts(amf, file_lut=FILE_LUT_GAS)

    n_wvl = len(wvl_mod)
    mr_gas_arr = mr_gas_arr / 1000. 
    delta_mr_ref = 1.
    k_spectre = calc_jac_rad(mr_gas_arr, n_wvl, t_gas_arr, delta_mr_ref)
    s = generate_filter(wvl_mod, wavelengths, fwhm)

    k = np.dot(k_spectre, s)
    return k


EXTENDED_WAVELENGTH_RANGE = (975, 2_445)
CLASSIC_WAVELENGTH_RANGE = (2_100, 2_445)
PERIODS_EXCLUDE_WATER = [(1260, 1330),]

def extended_bool_wavelengths(wavelengths:NDArray,
                              extended_wavelengths_range=EXTENDED_WAVELENGTH_RANGE)-> Tuple[np.array, np.array]:
    eb = (wavelengths >= extended_wavelengths_range[0]) & \
                                (wavelengths <= extended_wavelengths_range[1]) 
    
    for period in PERIODS_EXCLUDE_WATER:
        eb = eb & ~((wavelengths >= period[0]) & (wavelengths <= period[1]))

    return eb
    
def WMF_computation(emit_image:emit.EMITImage,
                     wavelengths, mean_vza, mean_sza, fwhm,
                     mask_water:bool=False, 
                     extended_wavelengths_range=EXTENDED_WAVELENGTH_RANGE,
                     classic_wavelengths_range=CLASSIC_WAVELENGTH_RANGE,
                     water_mask_threshold:float=0.05,
                     fill_value_default:float=-9999,
                     logger:Optional[logging.Logger]=None):
    """
    This function calculates wide matching filter proposed by Roger et al. 2023.
    The wide matching filter 

    Args:
        emit (EMITImage): EMITImage object
        wavelengths, mean_vza, mean_sza, fwhm,

        mask_water (bool, optional): If True, water pixels are masked. Defaults to False.
        water_mask_threshold (float, optional): Threshold to mask water pixels. Defaults to 0.05.
        fill_value_default (float, optional): Fill value for output products. Defaults to -9999.
    
    Returns:
        wide matching filter by column MF(975, 2_445) excluding the water absortion ranges: (1350-1420) and (1800, 1945) 
    """
    
    k_arr = load_target_spectrum_mf(wavelengths, mean_vza, mean_sza, fwhm)
    
    ex_bool_wv = extended_bool_wavelengths(wavelengths, extended_wavelengths_range=extended_wavelengths_range)

    indexes_extended = np.where(ex_bool_wv)[0]
    k_arr_extended = k_arr[indexes_extended]

    if logger is not None:
        logger.info(f"Extended wavelengths range: [{wavelengths[indexes_extended].min()}, {wavelengths[indexes_extended].max()}]")
    
    emit_image_extended_wavelengths = emit_image.read_from_bands(indexes_extended)

    indexes_classic_within_extended = np.where((wavelengths >= classic_wavelengths_range[0]) & \
                                                (wavelengths <= classic_wavelengths_range[1]))[0]
    if logger is not None:
        logger.info(f"Classic wavelengths range: [{emit_image_extended_wavelengths.wavelengths[indexes_classic_within_extended].min()}, {emit_image_extended_wavelengths.wavelengths[indexes_classic_within_extended].max()}]")
    
    img = emit_image_extended_wavelengths.load_raw(transpose=False) # (rows, cols, bands)
    img_masked = img.copy().astype(np.float64)
    img_masked[img_masked == fill_value_default] = np.nan
    if mask_water:
        # 2150-2440nm
        water_mask = np.any(img_masked[:,:,indexes_classic_within_extended] < water_mask_threshold, axis=-1) # (rows, cols)
        img_masked[water_mask,:] = np.nan

    mf_extended = AT_MF_select_window_alt(img_masked, k_arr_extended)
    mf_extended = np.nan_to_num(mf_extended, nan=fill_value_default)

    return mf_extended
