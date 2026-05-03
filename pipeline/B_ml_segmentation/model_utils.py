# Source https://github.com/UNEP-IMEO-MARS/marsml-hyperspectral
# Author: Vit Ruzicka

import numpy as np
import torch
import os

from utils.paths import codebase_folder
from utils.rio_utils import pad_sample, center_crop
from pipeline.B_ml_segmentation.model_handler import ModelHandler
from pipeline.B_ml_segmentation.fulltile_eval_utils import vectorize_plumemask, threshold_cutoff_connected_components
from utils.paths import models_storage
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
from georeader import rasterize, read
import geojson
import json
from shapely.geometry import Polygon, mapping


def HyperMARS_model_ensemble(settings, load_checkpoints = [""]):
    # Assuming the same model architecture for now...

    model_handlers = []
    for load_checkpoint in load_checkpoints:
        model_handler = HyperMARS_model(settings, load_checkpoint)
        model_handlers.append(model_handler)

    return model_handlers

def HyperMARS_model(settings, load_checkpoint = ""):
    # Return pytorch model object, ready for inference, weights loaded, potentially also training
    # - input: all setting variants
    # - output: pytorch model

    # MODEL
    model_handler = ModelHandler(settings)
    # model_handler.summarise()
    network = model_handler.get_network()

    if load_checkpoint != "":
        print("Loading model from", load_checkpoint)
        network.load_state_dict(torch.load(load_checkpoint, weights_only=True, map_location=model_handler.device))
        print("Weights loaded!")

    return model_handler

def HyperMARS_prepare_RGB_WMF(data_rgb, data_wmf, pad_to_multiple=32):
    # Input data processing
    # No-data
    data_wmf = np.nan_to_num(data_wmf, nan=0)
    data_wmf = np.where(data_wmf == -9999, 0, data_wmf)
    # Normalisation
    data_rgb = np.clip(data_rgb, -2, 2)
    data_wmf = np.clip(data_wmf, -2, 2)

    # Prepare into expected format
    x = np.concatenate((data_rgb, data_wmf), axis=0)
    assert len(x) == 4, f"expected 4 input channels, got {len(x)}"

    # valid mask (could also be loaded outside)
    valid_mask = data_rgb[0] != 0 # masks (1568, 1242)

    # Pad to multiple of 32
    _, orig_w, orig_h = x.shape
    # print("before padding", x.shape)
    x = pad_sample(x, pad_to_multiple, "constant")
    # print("after padding", x.shape)

    # To torch + batch dimension
    x = torch.Tensor(x)
    x = torch.unsqueeze(x, 0)
    assert len(x.shape) == 4, f"expected 4D tensor, got {x.shape}"

    return x, valid_mask, (orig_w, orig_h)

def HyperMARS_load_RGB_WMF_paths(rgb_path, wmf_path):
    wmf_geo = RasterioReader(wmf_path)
    rgb_geo = RasterioReader(rgb_path)
    return rgb_geo.values[0:3,:,:], wmf_geo.values, wmf_geo

def HyperMARS_ensemble_predict(models, inputs, masks, device, orig_shape, save_ensemble_uncertainties):
    predictions = []
    for model in models:
        prediction = HyperMARS_predict(model, inputs, masks, device, orig_shape)
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    ensemble_avg_prediction = np.mean(predictions, axis=0)

    if not save_ensemble_uncertainties:
        return ensemble_avg_prediction, None
    else:
        # Variance
        variance = np.var(predictions, axis=0) # variance_image
        return ensemble_avg_prediction, variance

def HyperMARS_predict(model, inputs, masks, device, orig_shape):
    # Place the data on the same device as the model
    x = inputs.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model.forward(x)

    # apply sigmoid (for most models ...)
    prediction = torch.sigmoid(prediction)

    # sanity check:
    assert (
            prediction.shape[2:] == x.shape[2:]
    ), f"prediction shape {prediction.shape} does not match input shape {x.shape}"

    orig_w, orig_h = orig_shape

    # Crop back into original resolution
    prediction_np = prediction[0][0].detach().cpu().numpy()
    prediction_cropped = center_crop(prediction_np, orig_w, orig_h)
    msg = f"expected output shape {orig_w}, {orig_h}, got {prediction_cropped.shape}"
    assert prediction_cropped.shape == (orig_w, orig_h), msg

    # print("prediction_cropped", prediction_cropped.shape)

    # Mask out prediction
    valid_mask = masks
    prediction_masked = prediction_cropped * valid_mask
    # print("prediction_masked", prediction_masked.shape)

    return prediction_masked

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def binary_prediction_to_vectors(prediction_continuous, reference_file, save_as = "tile_preds.geojson", tile_name="EMIT_L1B_RAD_001_20240629T051456_2418104_004", json_name="HyperMARS_model_pred"):

    initial_thr__for_detections = 0.5 # this should be 0.5 in most cases basically
    plumemask_initial_threshold_values = prediction_continuous > initial_thr__for_detections

    ref_data = RasterioReader(reference_file)

    plumemask_initial_threshold = GeoTensor(
        plumemask_initial_threshold_values,
        transform=ref_data.transform,
        crs=ref_data.crs,
        fill_value_default=False,
    )

    continuous_prediction = GeoTensor(
        prediction_continuous,
        transform=ref_data.transform,
        crs=ref_data.crs,
        fill_value_default=False,
    )

    minimal_instance_size = 5
    multipolygon = vectorize_plumemask(
        plumemask_initial_threshold, min_area=minimal_instance_size
    )

    # Compute the score for each plume
    scores = []
    for pol in multipolygon.geoms:
        # Subset the prediction to the plume polygon
        pred_cont_pol = read.read_from_polygon(
            continuous_prediction, pol, crs_polygon="EPSG:4326", pad_add=(1, 1)
        )
        plumemask_pol = rasterize.rasterize_geometry_like(
            pol, pred_cont_pol, crs_geometry="EPSG:4326", all_touched=True
        )

        # Set to zero the pixels outside the plume polygon
        pred_cont_pol = pred_cont_pol * plumemask_pol.astype(np.float32)

        score = threshold_cutoff_connected_components(
            pred_cont_pol, threshold_pixels=minimal_instance_size, tol=1e-3
        )
        scores.append(score)


    # ================
    # print("Saving vectors as:", save_as)
    polygons = multipolygon.geoms

    features = []
    for poly_i, poly in enumerate(polygons):
        geojson_dict = mapping(poly) # shapely.geometry.polygon.Polygon first to dictionary and then extract just the coords
        poly_coords = geojson_dict["coordinates"]

        geometry = geojson.Polygon(poly_coords)
        feature = geojson.Feature(geometry=geometry, properties={"confidence": scores[poly_i]})
        features.append(feature)

    # Create a FeatureCollection
    feature_collection = geojson.FeatureCollection(features)
    feature_collection["name"] = json_name

    # Include a start and a stop time (from granule name)
    # e.g.: "EMIT_L1B_RAD_001_20240629T051456_2418104_004"
    tile_datetime = tile_name.split("_")[4] # 20240629T051456
    from datetime import datetime, timedelta
    format_string = "%Y%m%dT%H%M%S" # 2024-06-29 05:14:56
    datetime_start = datetime.strptime(tile_datetime, format_string)
    one_second = timedelta(seconds=1)
    datetime_end = datetime_start + one_second

    format_out = '%Y-%m-%dT%H:%M:%SZ'
    feature_collection["datetime_start"] = datetime_start.strftime(format_out)
    feature_collection["datetime_end"] = datetime_end.strftime(format_out)

    # Save the FeatureCollection to a .geojson file
    with open(save_as, 'w') as f:
        json.dump(feature_collection, f, indent=2, cls=NumpyEncoder)

    # print(f"Polygons saved as FeatureCollection to {save_as}")


def load_single_model():
    import omegaconf

    # "/Users/ruzicka/Downloads/CODES/UN/marsml-hyperspectral/scripts/settings.yaml"
    settings_path = os.path.join(codebase_folder(), "parameters", "ml_model_settings", "settings.yaml")
    settings = omegaconf.OmegaConf.load(settings_path)

    # Example model
    load_checkpoint = os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R3_best_vf1_ep30.pt")
    model_handler = HyperMARS_model(settings, load_checkpoint)
    print("Model successfully loaded!")

    return model_handler

def load_ensemble_model():
    import omegaconf
    # "/Users/ruzicka/Downloads/CODES/UN/marsml-hyperspectral/scripts/settings.yaml"
    settings_path = os.path.join(codebase_folder(), "parameters", "ml_model_settings", "settings.yaml")
    settings = omegaconf.OmegaConf.load(settings_path)

    load_checkpoint = [
        os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R1_best_vf1_ep13.pt"),
        os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R2_best_vf1_ep30.pt"),
        os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R3_best_vf1_ep30.pt"),
        os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R4_best_vf1_ep50.pt"),
        os.path.join(models_storage(), "ch4_UNET_RGB_WMF", "UNET_RGB_WMF_R5_best_vf1_ep58.pt"),
    ]
    model_handlers = HyperMARS_model_ensemble(settings, load_checkpoints=load_checkpoint)

    print("Model ensemble successfully loaded!")

    return model_handlers
