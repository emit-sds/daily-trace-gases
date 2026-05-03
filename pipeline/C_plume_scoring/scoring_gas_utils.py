import numpy as np
from pipeline.C_plume_scoring.run_scoring import load_and_prep_data, run_plume_vetting_on_scene
from parameters.emit_waves import INSTRUMENT_WAVELENGTHS

def get_gas_settings(gas):
    ind_fits = []

    if gas == "co":
        mf_threshold = 300
        ind_out = list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 381) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1633))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1692) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2094))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2441) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2493)))
        ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2278) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2441))
        ind_fits.append(ind_fit)

    elif gas == "no2":
        mf_threshold = 300 * 4
        # OUT - same as methane, except start later
        ind_out = list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 753) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1633))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1692) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2094))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2441) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2493)))
        # IN (a) NO2: 0-50 (381-753)
        ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 381) & (np.array(INSTRUMENT_WAVELENGTHS) <= 753))
        ind_fits.append(ind_fit)

    elif gas == "nh3":
        mf_threshold = 300
        ind_out = list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 381) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1498))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1603) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1922))) + \
                  list(np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2441) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2493)))

        # (a) RANGE 1: 150 - 164: 1498-1603
        ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1498) & (np.array(INSTRUMENT_WAVELENGTHS) <= 1603))
        ind_fits.append(ind_fit)
        # (a) RANGE 2: 211 - 235: 1952-2130
        ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 1952) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2130))
        ind_fits.append(ind_fit)
        # (c) RANGE 3: 234 - 262: 2123-2326
        ind_fit = np.where((np.array(INSTRUMENT_WAVELENGTHS) >= 2123) & (np.array(INSTRUMENT_WAVELENGTHS) <= 2326))
        ind_fits.append(ind_fit)

    else:
        print("Gas", gas, "not implemented!")
        assert False

    return mf_threshold, ind_fits, ind_out

def get_score_for_tile(tile_id, raws_folder, results_folder, vectors, cmf_for_gas, gas, target_signature,
                                  min_polygon_size=0, debug_viz=False, debug_jump_to = None, preds_name = "prediction.geojson"):


    rdn, _, mask, _, cmf_path, scene_ch4_target_signature, vector_loaded_path = load_and_prep_data(tile_id, raws_folder=raws_folder, results_folder=results_folder, preds_name=preds_name, load_signature=False)

    mf_threshold, ind_fits, ind_out = get_gas_settings(gas)

    scores_per_polygon = []
    plotting_data_per_polygon = []

    # In most case run only once, however for NH3 we use three regions
    for dnorm_idx in range(len(ind_fits)):

        gas_settings = {}
        gas_settings["ind_fit"] = ind_fits[dnorm_idx]
        gas_settings["ind_out"] = ind_out

        scores_per_polygon_i, plotting_data_per_polygon_i = run_plume_vetting_on_scene(rdn, cmf_for_gas, mask, vectors, cmf_path, target_signature,
                         debug_viz=debug_viz, min_polygon_size=min_polygon_size, experimental_other_gases=gas,
                         other_gas_settings=gas_settings,
                         mf_threshold=mf_threshold,
                         debug_jump_to=debug_jump_to
        )
        scores_per_polygon.append(scores_per_polygon_i)
        plotting_data_per_polygon.append(plotting_data_per_polygon_i)

    scores_per_polygon_aggreg = {}

    num_of_dnorms = len(scores_per_polygon)
    keys = scores_per_polygon[0].keys()

    for k in keys:
        if len(scores_per_polygon) > 1:
            if k not in scores_per_polygon[1].keys(): continue
        if len(scores_per_polygon) > 2:
            if k not in scores_per_polygon[2].keys(): continue

        # print("Polygon", k, "sucessfully scored in all")
        number_of_pixels = scores_per_polygon[0][k]['number_of_pixels']

        scores_per_polygon_aggreg[k] = {}
        for dnorm_idx in range(num_of_dnorms):
            dnorm_i = scores_per_polygon[dnorm_idx][k]['D_norm']
            scores_per_polygon_aggreg[k]["D_norm_"+str(dnorm_idx+1)] = dnorm_i
        for dnorm_idx in range(num_of_dnorms):
            dnorm_i = scores_per_polygon[dnorm_idx][k]['alpha_con_len']
            scores_per_polygon_aggreg[k]["alpha"+str(dnorm_idx+1)] = dnorm_i
        scores_per_polygon_aggreg[k]["number_of_pixels"] = number_of_pixels

    # mostly return the DNORM 1 values for plots
    dnorm_plot_data = plotting_data_per_polygon[0]
    if gas == "nh3": # but for NH3 plot show the DNORM 2 fit
        dnorm_plot_data = plotting_data_per_polygon[1]

    return scores_per_polygon_aggreg, dnorm_plot_data
