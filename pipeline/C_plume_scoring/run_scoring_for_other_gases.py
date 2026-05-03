import os.path
import numpy as np
from utils.rio_utils import rio_load
from utils.vec_utils import vec_load
from pipeline.C_plume_scoring.scoring_gas_utils import get_score_for_tile
from pipeline.C_plume_scoring.run_scoring import annotate_and_save_vectors

def score_trace_gas(gas, tile_ID, target_file, results_folder, raws_folder):

    cmf_file_name = gas+"-cmf.tif"
    predictions_file_name = gas+"-prediction"

    cmf_file_path = os.path.join(results_folder, cmf_file_name)
    save_vectors_path = os.path.join(results_folder, predictions_file_name + ".geojson")
    save_scored_vectors_path = os.path.join(results_folder, predictions_file_name + "_scored.geojson")

    debug_viz = False
    debug_jump_to = None

    min_polygon_size = 0
    look_at_band = 0
    cmf_for_gas = rio_load(cmf_file_path)
    cmf_for_gas = cmf_for_gas[look_at_band]

    plumes_vector = vec_load(save_vectors_path)

    vectors = []
    for idx, row in plumes_vector.iterrows():
        p = row["geometry"]
        vectors.append(p)

    target_signature = np.loadtxt(target_file)
    target_signature = target_signature[:, 2]

    # helps fitting:
    if gas == "nh3": target_signature = target_signature * 25

    preds_name = gas + "-prediction.geojson"
    scores_per_polygon, _ = get_score_for_tile(tile_ID, raws_folder, results_folder,
                         vectors, cmf_for_gas, gas, target_signature=target_signature,
                         min_polygon_size=min_polygon_size, debug_viz=debug_viz, debug_jump_to=debug_jump_to, preds_name=preds_name)

    annotate_and_save_vectors(tile_ID, vectors, scores_per_polygon, save_vectors_path, save_scored_vectors_path, description="ModelPredsScored_" + gas)

    return save_scored_vectors_path
