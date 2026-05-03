import rasterio
import numpy as np
from skimage import measure
from georeader.geotensor import GeoTensor
from typing import Optional, Union
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from georeader.vectorize import get_polygons
from numpy.typing import NDArray

def binary_connected_prediction(pred_continuous_values, threshold_prediction, threshold_pixels):
    """
    Returns a binary prediction where the connected components with less than `threshold_pixels` pixels are removed.

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_prediction (float): threshold value for the prediction
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.

    Returns:
        Union[GeoTensor,NDArray]: binary prediction of type uint8 where the connected components with less than `threshold_pixels` pixels are removed.
    """
    pred_discrete = (pred_continuous_values > threshold_prediction).astype(np.uint8)
    labels, nclusters = measure.label(pred_discrete,
                                      connectivity=2, return_num=True)  # detect clusters and store their properties

    for cluster in range(1, nclusters + 1):
        labels_cluster = labels == cluster
        if np.sum(labels_cluster) < threshold_pixels:
            labels[labels_cluster] = 0

    pred_values = (labels > 0).astype(np.uint8)
    return pred_values


def count_connected_pixels(pred_continuous, threshold_prediction, threshold_pixels):
    """
    Counts the number of connected components in the scene with values above `threshold_prediction`.

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_prediction (float): threshold value for the prediction
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.

    Returns:
        int: number of connected components in the scene with values above `threshold_prediction`
    """
    binary_pred_values = binary_connected_prediction(pred_continuous, threshold_prediction, threshold_pixels)
    return int(np.sum(binary_pred_values))


def polygon_exterior(polygon: Polygon) -> Polygon:
    if len(list(polygon.interiors)) > 0:
        # keep the exterior
        geometry = Polygon(polygon.exterior.coords)
    else:
        geometry = polygon
    return geometry

def vectorize_plumemask(
        plumemask: GeoTensor,
        min_area: float = 25.5,
        footprint: Optional[Polygon] = None,
    ) -> MultiPolygon:
        """
        Function to vectorize the plume mask. Returns the MultiPolygon and a boolean indicating if the plume is empty.

        Args:
            plumemask (GeoTensor): GeoTensor with the plume mask.
            min_area (float, optional): Minimum area in pixels to consider a plume. Defaults to 25.5.
            footprint (Optional[Polygon], optional): remove polygons that do not intersect with the footprint. Defaults to None.

        Returns:
            MultiPolygon]: MultiPolygon with the plume (could be empty
        """
        pols_plume = []
        if np.any(plumemask.values):
            # self.logger.debug(f"Vectorizing plume mask for {image_to_process.tile}")
            plumemask_bool = plumemask.astype(bool)
            plumemask_bool.fill_value_default = False
            pols_plume = get_polygons(plumemask_bool, min_area=min_area)
        else:
            return MultiPolygon([])

        # convert pols to EPSG:4326
        if len(pols_plume) > 0:
            pols_plume = [
                shape(rasterio.warp.transform_geom(plumemask.crs, "EPSG:4326", mapping(p)))
                for p in pols_plume
            ]
        else:
            return MultiPolygon([])

        # remove polygons that do not intersect with the footprint.
        if footprint is not None:
            pols_plume = [p for p in pols_plume if p.intersects(footprint)]

        # Remove interior blobs, apply union to avoid invalid polygons
        if len(pols_plume) > 0:
            multiorpol = unary_union([polygon_exterior(p) for p in pols_plume])
            if isinstance(multiorpol, Polygon):
                pols_plume = [multiorpol]
            elif isinstance(multiorpol, MultiPolygon):
                pols_plume = list(multiorpol.geoms)
            else:
                raise ValueError(f"Geometry is not a Polygon or MultiPolygon {multiorpol}")
        else:
            return MultiPolygon([])

        return MultiPolygon(pols_plume)

def threshold_cutoff_connected_components(
    pred_continuous: Union[GeoTensor, NDArray],
    threshold_pixels: float,
    tol: float = 1e-3,
) -> float:
    """
    Implements binary search to find the continuous value that produces more than `threshold_pixels` pixels connected
    in the scene.

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.
        tol (float, optional): Tolerance for the binary search. Defaults to 1e-3.

    Returns:
        scene_prob (float): minimum value such that sum(connected_components(pred_continuous >= scene_prob)) >= threshold_pixels
    """
    if isinstance(pred_continuous, GeoTensor):
        pred_continuous_values = pred_continuous.values
    else:
        pred_continuous_values = pred_continuous

    min_value = np.min(pred_continuous_values)
    max_value = np.max(pred_continuous_values)

    # binary search
    threshold = (min_value + max_value) / 2
    while (max_value - min_value) > tol:

        npixels_connected = count_connected_pixels(
            pred_continuous_values, threshold, threshold_pixels=threshold_pixels
        )
        if npixels_connected >= threshold_pixels:
            min_value = threshold
        else:
            max_value = threshold
        threshold = (min_value + max_value) / 2

    return threshold

