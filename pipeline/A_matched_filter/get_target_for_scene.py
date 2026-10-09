import os
import pipeline.A_matched_filter.cmf_emit_ghg.spec_io as spec_io
import pipeline.A_matched_filter.cmf_emit_ghg.target_generation as target_generation
import numpy as np
import rasterio
from utils.download_utils import get_obs_name, get_mask_name
from utils.paths import lut_file_path

def generate_target_for_scene(radiance_file, target_file, obs_file, loc_file, mask_file, lut_file, gas="ch4"):
    product_version = "001"
    if "EMIT_L1B_RAD_002" in radiance_file:
        product_version = "002"

    if os.path.isfile(target_file) is False:
        _, obs = spec_io.load_data(obs_file)
        sza = obs[...,4]

        mean_sza = np.mean(sza[sza != -9999])

        _, elevation = spec_io.load_data(loc_file, return_loc_from_l1b_rad_nc=True)
        elevation = elevation[:,:,:2]
        mean_elevation = np.mean(elevation[elevation != -9999]) / 1000.
        mean_elevation = min(max(0, mean_elevation),3)

        # V001 of the RAD products:
        if product_version == "V001":
            if mask_file is not None:
                m_state, d_state = spec_io.load_data(mask_file, mask_type='mask')

                if mask_file.endswith('.nc'):
                    ind = m_state.band_names.index('H2O (g cm-2)')
                    mean_h2o = np.mean(d_state[:,:,ind])
                else:
                    ind = m_state.band_names.index('H2OSTR')
                    mean_h2o = np.mean(d_state[:,0,ind])
            else:
                # Just guess something...
                exit()
                mean_h2o = 1.3

        elif product_version == "002":
            # Where the water vapour is stored has changed between V001 and V002:
            rfl_name = str(mask_file).split("/")[-1].replace("_L2A_MASK_003_", "_L2A_RFL_002_")
            rfl_path = "/".join(str(mask_file).split("/")[:-1]) + "/" + rfl_name

            with rasterio.open(rfl_path) as src:
                with rasterio.open(src.subdatasets[-1]) as sub:
                    water_vapor = sub.read(1)

            # mask out -9999 no data values
            water_vapor = np.where(water_vapor == -9999.0, np.nan, water_vapor)
            mean_h2o = np.nanmean(water_vapor)
            print("mean_h2o", mean_h2o)

    # Run target generation
    if (os.path.isfile(target_file) is False):
        target_params = ['-z', str(mean_sza),
                         '-s', '100',
                         '-g', str(mean_elevation),
                         '-w', str(mean_h2o),
                         '--output', target_file,
                         '--hdr', radiance_file,
                         '--lut_dataset', lut_file]
        if gas == "co2":
            target_params.append('--co2')

        print("target_params", target_params)

        target_generation.main(target_params)


def ch4_target_for_scene(tile_ID, raws_folder, results_folder):
    # Generate a target signature file for this scene for ch4
    rad_name = tile_ID + ".nc"
    radiance_file = os.path.join(raws_folder, rad_name)
    obs_file = os.path.join(raws_folder, get_obs_name(tile_ID, with_file_type=True))
    loc_file = radiance_file
    mask_file = os.path.join(raws_folder, get_mask_name(tile_ID, with_file_type=True))
    target_file = os.path.join(raws_folder, "target_ch4_"+tile_ID)
    lut_file = lut_file_path()
    generate_target_for_scene(radiance_file, target_file, obs_file, loc_file, mask_file, lut_file, gas="ch4")

    return target_file