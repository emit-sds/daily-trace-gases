import os
from utils.download_utils import download_granule
from pipeline.A_matched_filter.wmf_utils.run_wmf import run_wmf
from utils.rio_utils import file_exists


def ch4_wmf_and_rgb_for_scene(tile_ID, raws_folder, results_folder):
    print("Downloading L1 data (RAD, OBS and mask) (... This might take some time!)")
    download_granule(tile_ID, raws_folder, also_download_l2a_mask=True, also_download_official_cmf=True)

    rad_name = tile_ID + ".nc"
    radiance_file = os.path.join(raws_folder, rad_name)

    # WMF + RGB
    results_wmf_path = os.path.join(results_folder, "ch4-wmf.tif")
    results_rgb_path = os.path.join(results_folder, "rgb.tif")

    if not file_exists(results_wmf_path):
        print("Cooking WMF and RGB products")
        run_wmf(results_wmf_path, results_rgb_path, radiance_file)

