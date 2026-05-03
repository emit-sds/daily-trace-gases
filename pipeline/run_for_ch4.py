### Fully Automatic Trace Gas Plume Detection - runner for ch4
### Author: Vit Ruzicka, 2026

from pipeline.A_matched_filter.run_wmf_for_scene import ch4_wmf_and_rgb_for_scene
from pipeline.B_ml_segmentation.run_model_on_scene import run_model_on_scene
from pipeline.C_plume_scoring.run_scoring import score_ch4

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
    print("----------------------------------------")
    print("Step 3: scoring predictions")
    saved_scored_vectors_path = score_ch4(tile_ID, results_folder, raws_folder, use_ensemble = True)

    return saved_scored_vectors_path


