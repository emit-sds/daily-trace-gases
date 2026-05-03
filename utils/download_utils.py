import os.path

import earthaccess
from earthaccess import download
# auth = earthaccess.login()
# example: earthaccess.download(["https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20240616T072726_2416805_025/EMIT_L2B_CH4ENH_002_20240616T072726_2416805_025.tif"], local_path=".")

from utils.rio_utils import file_exists

def get_rad_name(tile_name, with_file_type=False):
    if with_file_type: return tile_name+".nc"
    return tile_name
def get_obs_name(tile_name, with_file_type=False):
    name = tile_name.replace("_RAD_", "_OBS_")
    if with_file_type: return name+".nc"
    return name
def get_cmf_name(tile_name, with_file_type=False):
    name = tile_name.replace("_L1B_RAD_001_","_L2B_CH4ENH_002_")
    if with_file_type: return name+".tif"
    return name
def get_mask_name(tile_name, with_file_type=False):
    name = tile_name.replace("_L1B_RAD_", "_L2A_MASK_")
    if with_file_type: return name+".nc"
    return name

def download_granule(tile_name = "EMIT_L1B_RAD_001_20250529T024614_2514902_001", local_path = "granule_downloads",
                     download_l1b = True,
                     also_download_l2a_mask = False,
                     also_download_l2a_rfl = False,
                     also_download_official_cmf = False):
    rad_name = tile_name
    obs_name = tile_name.replace("_RAD_","_OBS_")

    links = []
    if download_l1b:
        links = ["https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/"+rad_name+"/"+rad_name+".nc",
          "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/"+rad_name+"/"+obs_name+".nc"]

    if also_download_l2a_mask:
        rfl_name = tile_name.replace("_L1B_RAD_","_L2A_RFL_")
        mask_name = tile_name.replace("_L1B_RAD_","_L2A_MASK_")
        l2a_rfl_link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/"+rfl_name+"/"+rfl_name+".nc"
        l2a_mask_link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/"+rfl_name+"/"+mask_name+".nc"
        if also_download_l2a_rfl: links.append(l2a_rfl_link)
        links.append(l2a_mask_link)

    if also_download_official_cmf:
        cmf_name = tile_name.replace("_L1B_RAD_001_","_L2B_CH4ENH_002_")
        cmf_link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/"+cmf_name+"/"+cmf_name+".tif"
        links.append(cmf_link)

    # check if these files already exist?
    file_needed = []
    if download_l1b: file_needed += [os.path.join(local_path, rad_name+".nc"), os.path.join(local_path, obs_name+".nc")]
    if also_download_l2a_mask:
        if also_download_l2a_rfl: file_needed.append(os.path.join(local_path, rfl_name + ".nc"))
        file_needed.append(os.path.join(local_path, mask_name + ".nc"))
    if also_download_official_cmf:
        file_needed.append(os.path.join(local_path, cmf_name + ".tif"))

    links_missing = []
    all_exist = True
    for idx, file in enumerate(file_needed):
        if not file_exists(file):
            links_missing.append(links[idx])
            all_exist = False
    if all_exist:
            print("Already dowloaded previously, skipping!")
            return True

    # Download only the missing ones:
    auth = earthaccess.login(persist=True)
    downloaded_list = earthaccess.download(links_missing, local_path=local_path)
    print("Downloading done!")
