### Fully Automatic Trace Gas Plume Detection - runner for other trace gases
### Author: Vit Ruzicka, 2026

import os.path
import shutil
from utils.paths import codebase_folder, models_storage
from utils.rio_utils import file_exists
from pipeline.A_matched_filter.run_cmf_for_target_signatures import run_cmf_using_target_file_tile_ID
from pipeline.B_ml_segmentation.run_models_other_gases import run_models_main_other_gases
from pipeline.C_plume_scoring.run_scoring_for_other_gases import score_trace_gas

def run_for_trace_gas(gas, tile_ID, results_folder, raws_folder):
    cmf_file_name = gas+"-cmf.tif"
    predictions_file_name = gas+"-prediction"
    signatures_folder = os.path.join(codebase_folder(), "parameters", "target_signatures")
    cmf_file_path = os.path.join(results_folder, cmf_file_name)
    save_prediction_path = os.path.join(results_folder, predictions_file_name + ".tif")
    save_vectors_path = os.path.join(results_folder, predictions_file_name + ".geojson")

    gases = {
        "nh3": "NH3_10000",
        "no2": "NO2_100",
        "co": "CO_1000",
    }
    if gas not in gases.keys():
        print("REQUESTED GAS=", gas, "not supported!")
        assert False
    gases_files = {
        "NO2_100": "EMIT_absorption_spectrum_NO2_100.0_direct_radiance.txt",
        "NH3_10000": "EMIT_absorption_spectrum_NH3_10000.0_direct_radiance.txt",
        "CO_1000": "EMIT_absorption_spectrum_CO_1000.0_direct_radiance.txt",
    }
    target_file = os.path.join(signatures_folder, gases_files[gases[gas]])

    # 1 GET CMF
    print("----------------------------------------")
    print("Step 1: getting data, computing CMF")
    if not file_exists(cmf_file_path):
        result_cmf_file = run_cmf_using_target_file_tile_ID(tile_ID, gases[gas], target_file, raws_folder)
        shutil.copy(result_cmf_file, cmf_file_path)

    # 2 MODEL PREDICTION
    print("----------------------------------------")
    print("Step 2: model prediction")

    load_checkpoint = [
        os.path.join(models_storage(), "trace_UNET_CMF", "UNET_CMF_R1_best_vf1_ep14.pt"),
        os.path.join(models_storage(), "trace_UNET_CMF", "UNET_CMF_R2_best_vf1_ep33.pt"),
        os.path.join(models_storage(), "trace_UNET_CMF", "UNET_CMF_R3_best_vf1_ep17.pt"),
        os.path.join(models_storage(), "trace_UNET_CMF", "UNET_CMF_R4_best_vf1_ep23.pt"),
    ]
    run_models_main_other_gases(gas=gas, variant="EnsembleCMF",
                                save_prediction_path=save_prediction_path,
                                save_vectors_path=save_vectors_path,
                                tile_name=tile_ID,
                                cmf_path=cmf_file_path,
                                load_checkpoint=load_checkpoint,
                                )

    # 3 SCORE PREDICTIONS
    print("----------------------------------------")
    print("Step 3: scoring predictions")
    saved_scored_vectors_path = score_trace_gas(gas, tile_ID, target_file, results_folder, raws_folder)

    return saved_scored_vectors_path