from utils.rio_utils import rio_load, pad_sample
import os.path
import numpy as np
import torch
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from georeader.save import save_cog
from utils.paths import codebase_folder

from pipeline.B_ml_segmentation.model_utils import HyperMARS_model, HyperMARS_model_ensemble
from pipeline.B_ml_segmentation.model_utils import HyperMARS_predict, HyperMARS_ensemble_predict, binary_prediction_to_vectors


GLOBAL_MEMORY_MODELS_PRELOADED = {}
USE_GLOBAL_MEMORY = True

def load_data(gas, cmf_path, look_at_band = 0, show = True, pad_data = True, pad_mask = False, verbose = True, no_data_value = 0.0):
    if gas == "nh3":
        NORM_BY = 15_000

    if gas == "no2":
        NORM_BY = 70_000

    if gas == "co":
        NORM_BY = 12_000

    cmf = rio_load(cmf_path)[look_at_band]

    valid_mask = np.where(cmf == -9999, 0, 1) # 1 where its valid
    cmf_orig = np.where(cmf == -9999, no_data_value, cmf)

    print("before scaling min/mean/max", np.min(cmf_orig), np.mean(cmf_orig), np.max(cmf_orig))
    cmf_scaled = cmf_orig / NORM_BY
    print("after scaling min/mean/max", np.min(cmf_scaled), np.mean(cmf_scaled), np.max(cmf_scaled))
    data = cmf_scaled

    # make sure it's float32
    data = data.astype(np.float32)
    orig_w, orig_h = data.shape

    data = np.expand_dims(data, axis=0)
    valid_mask = np.expand_dims(valid_mask, axis=0)

    if pad_data:
        if verbose: print("before padding:", data.shape)
        data = pad_sample(data, 32, "constant")
        if verbose: print("after padding:", data.shape)

    if pad_mask:
        valid_mask = pad_sample(valid_mask, 32, "constant")

    return data, valid_mask, orig_w, orig_h, cmf_path


def run_models_main_other_gases(gas = "nh3", variant = "SingleCMF", load_checkpoint = "", cmf_path = "",
                                save_prediction_path = "", save_vectors_path = "",
                                tile_name = "",
                                reproject_result = True,
    ):
    global GLOBAL_MEMORY_MODELS_PRELOADED
    global USE_GLOBAL_MEMORY

    print("Loading model weights and predicting")
    if variant == "SingleCMF" or variant == "EnsembleCMF":
        import omegaconf
        setting_path = os.path.join(codebase_folder(), "parameters", "ml_model_settings", "settings.yaml")
        settings = omegaconf.OmegaConf.load(setting_path)

        if variant == "EnsembleCMF" or variant == "SingleCMF":
            settings.model.num_channels = 1

        model_name = variant + "_"
        loaded_from_memory = False
        if USE_GLOBAL_MEMORY:
            if variant == "SingleCMF":
                if "model_single" in GLOBAL_MEMORY_MODELS_PRELOADED.keys():
                    model_handler = GLOBAL_MEMORY_MODELS_PRELOADED["model_single"]
                    model_name = GLOBAL_MEMORY_MODELS_PRELOADED["model_name"]
                    weights_name = GLOBAL_MEMORY_MODELS_PRELOADED["weights_name"]
                    loaded_from_memory = True
            if variant == "EnsembleCMF":
                if "model_ensemble" in GLOBAL_MEMORY_MODELS_PRELOADED.keys():
                    model_ensemble = GLOBAL_MEMORY_MODELS_PRELOADED["model_ensemble"]
                    model_name = GLOBAL_MEMORY_MODELS_PRELOADED["model_name"]
                    weights_name = GLOBAL_MEMORY_MODELS_PRELOADED["weights_name"]
                    loaded_from_memory = True

        if not loaded_from_memory:
            if variant == "SingleCMF":
                weights_name = load_checkpoint.split("/")[-1]
                model_handler = HyperMARS_model(settings, load_checkpoint)
                print("Model successfully loaded!")
                model_name += load_checkpoint.split("/")[-2]+"_"+load_checkpoint.split("/")[-1]

            if variant == "EnsembleCMF":
                load_checkpoints = load_checkpoint
                weights_name = "ensemble_of_" + str(len(load_checkpoints))
                model_ensemble = HyperMARS_model_ensemble(settings, load_checkpoints=load_checkpoints)
                print("Models (", len(model_ensemble), ") successfully loaded!")
                model_name += weights_name

            if USE_GLOBAL_MEMORY:
                if variant == "SingleCMF":
                    GLOBAL_MEMORY_MODELS_PRELOADED["model_single"] = model_handler
                    GLOBAL_MEMORY_MODELS_PRELOADED["model_name"] = model_name
                    GLOBAL_MEMORY_MODELS_PRELOADED["weights_name"] = weights_name
                if variant == "EnsembleCMF":
                    GLOBAL_MEMORY_MODELS_PRELOADED["model_ensemble"] = model_ensemble
                    GLOBAL_MEMORY_MODELS_PRELOADED["model_name"] = model_name
                    GLOBAL_MEMORY_MODELS_PRELOADED["weights_name"] = weights_name

        # Load CMF
        cmf_data, valid_mask, orig_w, orig_h, path_to_cmf = load_data(gas, cmf_path)
        cmf_data = torch.from_numpy(cmf_data)
        # C,W,H => B,C,W,H
        cmf_data = cmf_data.unsqueeze(0)
        valid_mask = valid_mask[0]

        if variant == "SingleCMF":
            print("Using device", model_handler.device)
            prediction = HyperMARS_predict(model_handler, cmf_data, valid_mask, model_handler.device, (orig_w, orig_h))
        if variant == "EnsembleCMF":
            print("Using device", model_ensemble[0].device)
            prediction, _ = HyperMARS_ensemble_predict(model_ensemble, cmf_data, valid_mask, model_ensemble[0].device, (orig_w, orig_h), save_ensemble_uncertainties=False)

    print("Prediction with shape:", prediction.shape)
    cmf_geo = RasterioReader(cmf_path)

    preds_fill_value_default = 0
    mask_predictions = True
    if mask_predictions:
        # mask no data values to -9999
        prediction_masked = np.where(valid_mask == 1, prediction, -9999)
        prediction = prediction_masked
        preds_fill_value_default = -9999

    geo_continuous_prediction = GeoTensor(
        prediction,
        transform=cmf_geo.transform,
        crs=cmf_geo.crs,
        fill_value_default=preds_fill_value_default,
    )
    save_cog(geo_continuous_prediction, save_prediction_path, descriptions=["predictions_" + weights_name],
             tags={"units": "model_prediction"})
    print("Saved prediction to", save_prediction_path, "!")

    binary_prediction_to_vectors(prediction, cmf_path, save_as=save_vectors_path, tile_name=tile_name, json_name=tile_name+"__"+model_name)
    print("Saved vectors to", save_vectors_path, "!")

