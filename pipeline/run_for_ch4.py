### Fully Automatic Trace Gas Plume Detection - runner for ch4
### Author: Vit Ruzicka, 2026

from pipeline.A_matched_filter.run_wmf_for_scene import ch4_wmf_and_rgb_for_scene
from pipeline.A_matched_filter.run_cmf_for_target_signatures import run_cmf_using_target_file_tile_ID
from pipeline.A_matched_filter.get_target_for_scene import ch4_target_for_scene
from pipeline.B_ml_segmentation.run_model_on_scene import run_model_on_scene
from pipeline.C_plume_scoring.run_scoring import score_ch4
from utils.rio_utils import file_exists
import os
import shutil
from utils.download_utils import download_granule

def run_for_ch4(tile_ID, results_folder, raws_folder):

    # 1 GET AND COMPUTE DATA (WMF, RGB)
    print("----------------------------------------")
    print("Step 1: getting data, computing WMF, RGB")
    ch4_wmf_and_rgb_for_scene(tile_ID, raws_folder, results_folder)

    # 2 MODEL PREDICTION
    print("----------------------------------------")
    print("Step 2: model prediction")
    run_model_on_scene(tile_ID, results_folder, use_ensemble = True)

    # 3 SCORE PREDICTIONS
    if "EMIT_L1B_RAD_002_" in tile_ID:
        # special case - V002 radiances don't have the CH4ENH (MF) product available, instead let's cook it here ourselves:
        print("----------------------------------------")
        print("Step 3prep: for V002 we also need to prepare CMF ourselves, it used during scoring")
        cmf_file_name = tile_ID.replace("_L1B_RAD_002_", "_L2B_CH4ENH_003_")+".tif"
        cmf_file_path = os.path.join(raws_folder, cmf_file_name) # this is where the later scoring code searches for it ...

        if not file_exists(cmf_file_path):
            download_granule(tile_ID, raws_folder, also_download_l2a_rfl=True, also_download_l2a_mask=True) # < downloads only the RFL now
            # TODO: possibly only download the WV band that we later need

            target_file = ch4_target_for_scene(tile_ID, raws_folder, results_folder)
            result_cmf_file = run_cmf_using_target_file_tile_ID(tile_ID, "ch4", target_file, raws_folder)
            shutil.copy(result_cmf_file, cmf_file_path)

    print("Step 3: scoring predictions")
    saved_scored_vectors_path = score_ch4(tile_ID, results_folder, raws_folder, use_ensemble = True)

    return saved_scored_vectors_path


