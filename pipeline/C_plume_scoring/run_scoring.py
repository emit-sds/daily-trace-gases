import os.path
import numpy as np
import geojson
import json

from utils.download_utils import get_cmf_name, get_rad_name, get_mask_name
from utils.paths import codebase_folder
from utils.rio_utils import rio_load
from utils.vec_utils import vec_load
from utils.vec_utils import how_many_pixels_does_polygon_occupy
from utils.rio_utils import NCImage
from pipeline.C_plume_scoring.pv_utils.plume_vetting_utils import compute_masks, coords_inside_plume_from_binary
from pipeline.C_plume_scoring.pv_utils.sfun import get_pairs, get_ratio, calculate_fit
from parameters.emit_waves import INSTRUMENT_WAVELENGTHS

def load_and_prep_data(tile_ID, raws_folder, results_folder, preds_name = "prediction_ensemble.geojson", load_signature=True):
    predictions_path = os.path.join(results_folder, preds_name)
    rdn_path = os.path.join(raws_folder, get_rad_name(tile_ID, with_file_type=True))
    mask_path = os.path.join(raws_folder, get_mask_name(tile_ID, with_file_type=True))
    cmf_path = os.path.join(raws_folder, get_cmf_name(tile_ID, with_file_type=True))

    # Load data:
    # === L1B RAD data loading: ===
    ei = NCImage(rdn_path)
    data = ei.load(as_reflectance=False)

    rdn_ = data.values
    rdn_ = np.transpose(rdn_, axes=(1, 2, 0)) # Ch, W, H => i want into W,H,Ch again
    rdn_masked = np.where(rdn_ == -9999, 0, rdn_) # masked with 0 is close to how it's loaded in the main script ...
    rdn = rdn_masked

    # === L2B MF data loading: ===
    cmf_data = rio_load(cmf_path)[0]

    # === L2A mask loading: ===
    mask = NCImage(mask_path, layer_name = "mask", skip_wavelenghts=True)
    mask = mask.load(as_reflectance=False)
    mask_data = mask.values
    mask_data = np.transpose(mask_data, axes=(1, 2, 0))
    clouds_and_surface_water_mask = np.sum(mask_data[..., :3], axis=-1) > 0

    # === scene specific target signature ===
    scene_target_signature = None

    # This is only for demo results - for official reproductions we suggest using the official CMFs released for methane
    if load_signature:
        # Either load from CH4_static (which is the target computed for EMIT_L1B_RAD_001_20260422T114357_2611207_012)
        target_path = os.path.join(codebase_folder(), "parameters", "target_signatures", "EMIT_absorption_spectrum_CH4_static_direct_radiance.txt")
        ch4_target = np.loadtxt(target_path)
        scene_target_signature = ch4_target[:, 2]

        # Or just use a fixed target:
        # # For simplicity (and demo purposes) let's just use fixed target signature for methane ...
        # # Otherwise please refer to the codebase on https://github.com/emit-sds/emit-ghg/blob/main/target_generation.py
        # scene_target_signature = [9.65677880e-06, 9.44929334e-06, 9.04269344e-06, 8.85827101e-06, 8.98958035e-06, 8.49728372e-06, 8.62805714e-06, 8.80114715e-06, 8.33883158e-06, 8.16142819e-06, 7.73368404e-06, 7.39366692e-06, 7.35562982e-06, 7.04368029e-06, 6.11562144e-06, 6.53218762e-06, 6.63357218e-06, 5.78389142e-06, 5.84881855e-06, 5.47753266e-06, 4.79791781e-06, 1.13047456e-06, -2.35286791e-06, 4.59439270e-06, 4.56166689e-06, 3.33127292e-06, 1.54112465e-06, 3.26023549e-06, 3.48084766e-06, 2.07768389e-06, 1.34280623e-06, -1.43228457e-05, -5.30431391e-05, -1.43933273e-05, 1.97451749e-06, 2.04592925e-06, 1.06639434e-06, -5.74711749e-06, -1.36636143e-05, -8.58726232e-06, -5.89627356e-07, -2.22065616e-06, -3.23409978e-06, -2.62903649e-05, -2.16494465e-05, -5.92926997e-05, -2.94537249e-04, -3.05365354e-04, -6.61797152e-05, -3.59345933e-06, -9.58845533e-08, -6.98856370e-06, -2.20578091e-05, -4.63919870e-05, -9.98187735e-05, -1.06221661e-04, -1.34282159e-04, -8.69059866e-05, -3.41956800e-05,
        #               -7.52936476e-06, -1.03710475e-06, -4.22974056e-05, -8.64636580e-05, -7.36054663e-05, -3.05240747e-04, -5.80721283e-04, -5.79022997e-04, -3.15508683e-03, -7.20142290e-03, -5.69085827e-03, -2.20352407e-03, -4.58517744e-04, -1.39371442e-04, -3.95658213e-05, -1.05974738e-05, -6.18653290e-06, -3.43563370e-05, -6.65568568e-05, -2.57255106e-04, -5.96432718e-04, -6.37708274e-04, -1.44363145e-03, -1.96407999e-03, -1.44413259e-03, -1.86983579e-03, -1.77950763e-03, -7.16888202e-04, -4.00458836e-04, -2.25835846e-04, -1.02706943e-04, -2.08844006e-05, -1.18141154e-07, 5.29685862e-07, -1.92552801e-07, -6.41317179e-06, -1.58942158e-04, -2.15882719e-04, -4.84205933e-04, -2.26379462e-03, -6.85842307e-03, -1.50126939e-02, -1.57474495e-02, -1.18301261e-02, -2.05601913e-02, -3.60834425e-02, -3.94273630e-02, -2.59847522e-02, -1.44271172e-02, -9.88838744e-03, -6.08667109e-03, -3.18633724e-03, -1.46129171e-03, -5.25151565e-04, -2.60101417e-04, -3.37112286e-04, -1.97430909e-04,
        #               -8.00361556e-05, -1.10455109e-05, -1.31887203e-05, -4.19767046e-05, -3.32164105e-05, -2.49744551e-05, -7.15452259e-05, -3.87402691e-04, -2.02328613e-03, -7.70922092e-03, -1.23149560e-02, -2.02294852e-02, -2.76472263e-02, -2.87585064e-02, -2.83417192e-02, -3.16582578e-02, -2.93610347e-02, -3.23032725e-02, -3.81929431e-02, -3.90207197e-02, -3.29549597e-02, -3.48682336e-02, -3.12618589e-02, -2.05089329e-02, -1.49367305e-02, -1.07353593e-02, -5.97769992e-03, -4.02974442e-03, -2.81336234e-03, -2.08011794e-03, -2.54418498e-03, -2.72301685e-03, -1.72132735e-03, -9.38626716e-04, -5.69493323e-04, -4.56683372e-04, -3.41620643e-04, -2.17749254e-04, -1.29124440e-04, -9.29762234e-05, -6.92330156e-05, -5.67021771e-05, -6.96139050e-05, -6.38932674e-05, -4.11656578e-05, -3.67605277e-05, -7.57429864e-05, -2.71394567e-04, -1.01682139e-03, -3.75419836e-03, -1.26479782e-02, -3.75491237e-02, -8.38793133e-02, -1.13786173e-01, -1.16849649e-01, -8.59692033e-02, -1.38584499e-01,
        #               -1.44295087e-01, -6.46457727e-02, -7.58295710e-02, -8.64002790e-02, -1.05684196e-01, -1.23224796e-01, -6.18103379e-02, -8.05863781e-02, -1.00805092e-01, -6.55649471e-02, -4.37739533e-02, -3.14234067e-02, -3.00320311e-02, -4.31862871e-02, -6.02494192e-02, -4.14973363e-02, -7.14254345e-02, -5.12582042e-02, -2.41301424e-02, -2.49192260e-02, -2.28203546e-02, -1.85601865e-02, -1.44834816e-02, -1.14326348e-02, -8.08636595e-03, -6.13045088e-03, -4.44505264e-03, -2.75363038e-03, -1.76782748e-03, -1.29662555e-03, -1.13509987e-03, -1.27273834e-03, -1.68181124e-03, -2.11492102e-03, -2.41104042e-03, -1.42739322e-03, -9.57903434e-04, -1.20352719e-03, -1.00580261e-03, -6.22048310e-04, -6.37473072e-04, -5.43213916e-04, -5.01025722e-04, -3.52837832e-04, -2.48386828e-04, -2.06771481e-04, -1.71915054e-04, -1.75783825e-04, -9.43188133e-05, -7.43165031e-05, -4.79766004e-05, -4.43891285e-05, -2.59220423e-05, -2.82430025e-05, -4.60777162e-05, -6.02059693e-05, -6.53891201e-05,
        #               -9.07074440e-05, -1.41093437e-04, -2.96758124e-04, -7.87594072e-04, -2.17192119e-03, -5.58628668e-03, -1.28307233e-02, -2.48689824e-02, -4.44926900e-02, -6.73228909e-02, -7.92775824e-02, -7.57375261e-02, -5.13581699e-02, -3.93471604e-02, -2.78321330e-01, -3.55763275e-01, -9.97541568e-02, -1.58274724e-01, -2.33960939e-01, -3.49164500e-01, -4.32103636e-01, -5.05408477e-01, -5.59860848e-01, -6.24071005e-01, -5.76580213e-01, -5.90172912e-01, -6.55077760e-01, -8.45815529e-01, -9.15431133e-01, -5.23962589e-01, -7.43168727e-01, -7.78706093e-01, -5.71164846e-01, -8.73710217e-01, -1.14039687e+00, -1.03317436e+00, -6.33479317e-01, -8.39119315e-01, -1.10115544e+00, -5.98434457e-01, -5.94363617e-01, -5.53028991e-01, -4.19743280e-01, -2.49515472e-01, -2.23981385e-01, -2.46002450e-01, -1.57568885e-01, -1.20763738e-01, -1.00731546e-01, -7.19952086e-02, -4.84661609e-02, -3.50186769e-02, -2.38227621e-02, -2.20361438e-02, -1.95619810e-02]
        scene_target_signature = np.asarray(scene_target_signature)


    # === Vectors loading: ===
    print("loading vector from", predictions_path)
    plumes_vector = vec_load(predictions_path)
    plume_polygons = []
    for idx, row in plumes_vector.iterrows():
        p = row["geometry"]
        plume_polygons.append(p)

    print("Loaded data for ["+tile_ID+"]: L1A:", rdn.shape, "CMF:", cmf_data.shape, "cloud mask:", clouds_and_surface_water_mask.shape, "and", len(plume_polygons), " polygons!")
    return rdn, cmf_data, clouds_and_surface_water_mask, plume_polygons, cmf_path, scene_target_signature, predictions_path

def compute_scores(sig, ratio, deg_poly, wl_only_in_fit):
    coef = calculate_fit(deg_poly, wl_only_in_fit, sig, ratio)  # fit the model to measurement

    optimized_alpha = coef[0]
    polyn = np.polyval(coef[1:], wl_only_in_fit)
    fitsig = polyn * np.exp(optimized_alpha * sig)

    optimized_alpha *= 1e5  # multiply estimated concentration length with the scaling factor
    ratio_p = ratio / polyn
    fitsig_p = fitsig / polyn

    # calculate_magnitude(ratio_p, 1)
    mag_over_bands = np.abs(ratio_p - np.mean(ratio_p))
    mag_total = np.mean(mag_over_bands)  # <^ mean absolute deviation from the mean of the ratio.

    # calculate_dist(ratio_p, sig_p, 0) - first part
    d_norm_over_bands = np.abs(ratio_p - fitsig_p)

    # form of normalisation
    d_norm_over_bands = d_norm_over_bands / mag_total

    # calculate_dist(ratio_p, sig_p, 0) - second part
    d_norm_total = np.mean(d_norm_over_bands)

    return d_norm_total, optimized_alpha, ratio_p, fitsig_p, mag_total, fitsig, polyn

def annotate_and_save_vectors(tile_name, vectors, scores_per_polygon, vectors_json_path, save_as, description = "__HyperMARS_ensemble_of_5_SCORED"):

    plumes_vector = vec_load(vectors_json_path)
    plume_polygons = []
    for idx, row in plumes_vector.iterrows():
        p = {}
        for key in row.keys():
            p[key] = row[key]
        plume_polygons.append(p)

    for pol_idx, polygon in enumerate(vectors):
        if pol_idx in scores_per_polygon.keys():
            for metric_key in scores_per_polygon[pol_idx].keys():
                plume_polygons[pol_idx][metric_key] = scores_per_polygon[pol_idx][metric_key]

    # Save vector with updated values...
    json_name = tile_name + description

    features = []
    for poly_i, poly in enumerate(plume_polygons):
        geometry = poly["geometry"]
        properties = {}
        for metric_key in poly.keys():
            if metric_key == "geometry": continue
            properties[metric_key] = poly[metric_key]
        feature = geojson.Feature(geometry=geometry, properties=properties)
        features.append(feature)

    # Create a FeatureCollection
    feature_collection = geojson.FeatureCollection(features)
    feature_collection["name"] = json_name

    # Include a start and a stop time (from granule name)
    # "EMIT_L1B_RAD_001_20240629T051456_2418104_004"
    # print(tile_name.split("_"))
    tile_datetime = tile_name.split("_")[4]  # 20240629T051456
    from datetime import datetime, timedelta
    format_string = "%Y%m%dT%H%M%S"  # 2024-06-29 05:14:56
    datetime_start = datetime.strptime(tile_datetime, format_string)
    one_second = timedelta(seconds=1)
    datetime_end = datetime_start + one_second

    format_out = '%Y-%m-%dT%H:%M:%SZ'
    feature_collection["datetime_start"] = datetime_start.strftime(format_out)
    feature_collection["datetime_end"] = datetime_end.strftime(format_out)

    # Save the FeatureCollection to a .geojson file
    with open(save_as, 'w') as f:
        json.dump(feature_collection, f, indent=2)

    print(f"Polygons saved as FeatureCollection to {save_as}")


def run_plume_vetting_on_scene(rdn, cmf, mask, vectors, cmf_path, scene_target_signature=None, debug_viz=True,
                               num_pts=40, min_polygon_size=0, mf_threshold=30, debug_jump_to=None,
                               experimental_other_gases=None,
                               other_gas_settings=None):
    # HYPERPARAMS:
    radius = 200  # how far from the plume center do we look for points?
    deg_poly = 10  # degree of polynomial for the theoretical model # (paper maybe mentioned 6?)
    dist_opt = 1

    scores_per_polygon = {}
    plotting_data_per_polygon = {}

    for pol_idx, polygon in enumerate(vectors):
        if debug_jump_to is not None:
            if pol_idx != debug_jump_to: continue

        number_of_pixels, plume_mask = how_many_pixels_does_polygon_occupy(polygon, cmf_path)  # or wmf_path

        if number_of_pixels >= min_polygon_size:
            title = "Polygon #" + str(pol_idx) + " (" + str(number_of_pixels) + "px)"
            print(title)

            # Start with default reject values ...
            scores_per_polygon[pol_idx] = {"D_norm": 1.,
                                           "alpha_con_len": 0.,
                                           "number_of_pixels": int(number_of_pixels)
            }

            combined_mask, background_mask = compute_masks(cmf, mf_threshold, clouds_and_surface_water_mask=mask)
            orig_points_inside_plume = coords_inside_plume_from_binary(plume_mask)

            if experimental_other_gases is None:
                # CH4:

                ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2100) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2440))
                wl_only_in_fit = np.array(INSTRUMENT_WAVELENGTHS)[ind_fit]

                # indices of wavelenghts outside of methane visibility
                # from - to: 0 (381) - 168 (1632.8513), 176 (1692.358) - 230 (2093.3562), 277 (2441.2183) - 284 (2492.9238)
                ind_out = list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 381) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1633))) + \
                          list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1692) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2094))) + \
                          list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2441) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2493)))
                ind_out = np.hstack(ind_out)

            else:
                # OTHER THAN CH4
                if other_gas_settings is not None:
                    # print("OVERRIDE USING SETTINGS FROM OUTSIDE")
                    if "ind_fit" in other_gas_settings.keys():
                        ind_fit = other_gas_settings["ind_fit"]
                        wl_only_in_fit = np.array(INSTRUMENT_WAVELENGTHS)[ind_fit]

                    if "ind_out" in other_gas_settings.keys():
                        ind_out = other_gas_settings["ind_out"]
                        ind_out = np.hstack(ind_out)

            top_pairs = get_pairs(radius, num_pts, rdn, cmf, orig_points_inside_plume, ind_out, combined_mask, background_mask, dist_opt)

            if top_pairs is None: continue  # reject
            if len(top_pairs) == 0: continue  # reject

            ratio = get_ratio(rdn, top_pairs)
            sig = np.array(scene_target_signature)[ind_fit]
            ratio = np.array(ratio)[ind_fit]  # target-to-background radiance ratio
            d_norm, alpha, ratio_p, fitsig_p, mag_total, fitsig, polyn = compute_scores(sig, ratio, deg_poly, wl_only_in_fit)

            plotting_data_per_polygon[pol_idx] = {
                "ratio": ratio,
                "fit_sig": fitsig,
                "wl": wl_only_in_fit,
                "polyn": polyn,
            }
            scores_per_polygon[pol_idx] = {
                "D_norm": d_norm,
                "alpha_con_len": alpha,
                "number_of_pixels": int(number_of_pixels)
            }

    return scores_per_polygon, plotting_data_per_polygon

def score_ch4(tile_ID, results_folder, raws_folder, use_ensemble = True):
    preds_name = "prediction_single.geojson"
    scored_preds_name = "prediction_single_scored.geojson"

    if use_ensemble:
        preds_name = "prediction_ensemble.geojson"
        scored_preds_name = "prediction_ensemble_scored.geojson"

    predictions_path = os.path.join(results_folder, preds_name)
    scored_predictions_path = os.path.join(results_folder, scored_preds_name)

    rdn, cmf, mask, vectors, cmf_path, scene_target_signature, _ = load_and_prep_data(tile_ID, raws_folder, results_folder, preds_name=preds_name)

    print("Scoring vectors")
    scores_per_polygon, plotting_data_per_polygon = run_plume_vetting_on_scene(rdn, cmf, mask, vectors, cmf_path, scene_target_signature)

    print("Saving scored vectors")
    annotate_and_save_vectors(tile_ID, vectors, scores_per_polygon, predictions_path, scored_predictions_path, description="ModelPredsScored_ch4")

