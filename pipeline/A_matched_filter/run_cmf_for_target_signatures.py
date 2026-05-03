import os
from utils.download_utils import download_granule
from utils.paths import codebase_folder
import rasterio

import pipeline.A_matched_filter.cmf_emit_ghg.mf as mf
import pipeline.A_matched_filter.cmf_emit_ghg.spec_io as spec_io
from pipeline.A_matched_filter.cmf_emit_ghg.s_utils import convert_to_cog


# Below is taken from ghg_process...
def ortho_with_spec_io(obs_for_glt_filename, unortho_input_filename, ortho_output_filename):
    m_obs, _ = spec_io.load_data(obs_for_glt_filename, load_glt=True)
    m, d = spec_io.load_data(unortho_input_filename)
    d_ort = spec_io.ortho_data(d, m_obs.glt)
    m.geotransform = m_obs.geotransform
    m.projection = m_obs.projection
    spec_io.write_envi_file(d_ort, m, ortho_output_filename)


def run_cmf_using_target_file(gas_name, folder_path, radiance_file, glt_file, target_file, l2a_mask_file,
                              wavelength_range_override = None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    product_name = "cmf"
    mf_file = os.path.join(folder_path, product_name+"_product_"+gas_name)
    mf_ort_file = os.path.join(folder_path, product_name+"_product_"+gas_name+"_ort")
    mf_ort_cog = os.path.join(folder_path, product_name+"_product_"+gas_name+"_ort.tif")
    l1b_bandmask_file = l2a_mask_file
    flare_file = os.path.join(folder_path, product_name+"_product_flares.json")
    noise_file = os.path.join(codebase_folder(), "parameters", "instrument_noise_parameters", "emit_noise.txt")
    mf_sens_file = os.path.join(folder_path, product_name+"_product_sens")
    mf_uncert_file = os.path.join(folder_path, product_name+"_product_mf_uncert")

    wavelength_range = ['500', '1340', '1500', '1790', '1950', '2450']

    if wavelength_range_override is not None:
        wavelength_range = wavelength_range_override
        print("OVERRIDE, setting wavelength_range_override=", wavelength_range_override)

    overwrite = True

    if (os.path.isfile(mf_file) is False or overwrite):
        subargs = [radiance_file,
                   target_file,
                   mf_file,
                   '--n_mc', '1',
                   '--l1b_bandmask_file', l1b_bandmask_file,
                   '--l2a_mask_file', l2a_mask_file,
                   '--fixed_alpha', '0.0000000001',
                   '--flare_outfile', flare_file,
                   '--noise_parameters_file', noise_file,
                   '--sens_output_file', mf_sens_file,
                   '--uncert_output_file', mf_uncert_file]

        if wavelength_range is not None:
            subargs.extend(['--wavelength_range'] + [str(val) for val in wavelength_range])

        subargs.append('--mask_flares')
        subargs.append('--mask_clouds_water')

        override = {}

        mf.main(subargs, override=override)

        # ORT MF
        ortho_with_spec_io(glt_file, mf_file, mf_ort_file)
        convert_to_cog(mf_ort_file,
                       mf_ort_cog,
                       {
                           'name': 'EMIT_CMF',
                           'description': 'CMF gas - '+gas_name,
                           'units': 'ppm m'
                       },
                       None,
                       None)
        return mf_ort_cog

def run_cmf_using_target_file_tile_ID(tile_ID, gas_name, target_file, raws_folder,
                                      wavelength_range_override=None):

    print("Downloading L1 data (RAD, OBS and mask (and reference CMF)) (... This might take some time!)")
    download_granule(tile_ID, raws_folder, also_download_l2a_mask=True, also_download_official_cmf=True)

    rad_name = tile_ID + ".nc"
    obs_name = rad_name.replace("_RAD_","_OBS_")
    mask_name = rad_name.replace("_L1B_RAD_", "_L2A_MASK_")

    radiance_file = os.path.join(raws_folder, rad_name)
    obs_file = os.path.join(raws_folder, obs_name)
    l2a_mask_file = os.path.join(raws_folder, mask_name)

    print("Cooking CMF products")
    result_file = run_cmf_using_target_file(gas_name, raws_folder, radiance_file, obs_file, target_file, l2a_mask_file,
                                            wavelength_range_override=wavelength_range_override)

    # Geotiff file compression
    with rasterio.open(result_file) as src:
        profile = src.profile.copy()
        data = src.read(1)
        profile.update(driver='GTiff', count=1, compress='lzw')
    with rasterio.open(result_file, 'w', **profile) as dst:
        dst.write(data, 1)

    return result_file

