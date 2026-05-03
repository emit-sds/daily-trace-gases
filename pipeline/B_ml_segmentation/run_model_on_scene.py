from pipeline.B_ml_segmentation.model_utils import HyperMARS_prepare_RGB_WMF, HyperMARS_load_RGB_WMF_paths
from pipeline.B_ml_segmentation.model_utils import HyperMARS_predict, HyperMARS_ensemble_predict
from pipeline.B_ml_segmentation.model_utils import load_single_model, load_ensemble_model
from pipeline.B_ml_segmentation.model_utils import binary_prediction_to_vectors
import os
import numpy as np
from georeader.geotensor import GeoTensor
from georeader.save import save_cog

def run_model_on_scene(tile_ID, results_folder, use_ensemble = True):
    wmf_path = os.path.join(results_folder, "ch4-wmf.tif")
    rgb_path = os.path.join(results_folder, "rgb.tif")
    save_prediction_path = os.path.join(results_folder, "prediction_single.tif")
    save_vectors_path = os.path.join(results_folder, "prediction_single.geojson")

    if use_ensemble:
        save_prediction_path = os.path.join(results_folder, "prediction_ensemble.tif")
        save_vectors_path = os.path.join(results_folder, "prediction_ensemble.geojson")

    print("Loading and formatting model inputs")
    data_toa_radiances, data_wmf, geo = HyperMARS_load_RGB_WMF_paths(rgb_path, wmf_path)
    inputs, masks, orig_shape = HyperMARS_prepare_RGB_WMF(data_toa_radiances, data_wmf, pad_to_multiple=32)

    print("Loading model weights and predicting")
    if use_ensemble:
        model_ensemble = load_ensemble_model()

        device = model_ensemble[0].device
        prediction, ensemble_uncertainties = HyperMARS_ensemble_predict(model_ensemble, inputs, masks, device, orig_shape, save_ensemble_uncertainties=False)

    else:
        model = load_single_model()

        device = model.device
        prediction = HyperMARS_predict(model, inputs, masks, device, orig_shape)

    print("Prediction with shape:", prediction.shape)
    # mask no data values to -9999
    prediction_masked = np.where(masks == 1, prediction, -9999)
    prediction = prediction_masked
    preds_fill_value_default = -9999

    geo_continuous_prediction = GeoTensor(
        prediction,
        transform=geo.transform,
        crs=geo.crs,
        fill_value_default=preds_fill_value_default,
    )

    save_cog(geo_continuous_prediction, save_prediction_path, descriptions=["model_prediction"],
             tags={"units": "model_prediction"})
    print("Saved prediction to", save_prediction_path, "!")

    binary_prediction_to_vectors(prediction, reference_file=wmf_path, save_as=save_vectors_path,
                                 tile_name=tile_ID, json_name=tile_ID + "__" + "prediction")

    print("Saved vectors to", save_vectors_path, "!")
    return save_vectors_path

