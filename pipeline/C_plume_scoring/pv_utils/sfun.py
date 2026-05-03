# SOURCE > https://github.com/emit-sds/plume-vetting/blob/main/sfun.py
from functools import partial
import numpy as np
import scipy
from scipy.spatial.distance import cdist

def fit_exponential(wavelengths, epsilon, *coeffs, ac):
    L0 = np.polyval(coeffs, wavelengths)  # < prepared as polynomial function
    # v exponential transformation to the function
    # ... (matches the formula for Beer-Lambert Law L = L0 e ^ (-alpha * L)
    return L0 * np.exp(epsilon * ac)


def calculate_fit(degree, wl, sig, ratio):
    # degree of polynom = 10
    # wl, wavelengths = [2100.769 ... 2433.8303]
    # sig - signal of methane computed for this scene
    # ratio - ratio from in/out of plume points
    epsilon_initial = 0.01
    poly_initial = [1.0] + [0.0] * degree
    initial_guess = [epsilon_initial] + poly_initial

    lower_bounds = [0] + [-np.inf] * (degree + 1)
    upper_bounds = [1] + [np.inf] * (degree + 1)

    # use partial to fix ac while calling fit_exponential
    fit_func = partial(fit_exponential, ac=sig)

    coef, _ = scipy.optimize.curve_fit(f=fit_func, xdata=wl, ydata=ratio, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
    # ^ Use least squares to fit a function, f, to data.
    # polynomial approximation of exponential function (which comes from the fact that it's Beer-Lambert)
    return coef

def get_neighboring_points(top_points, square_size, point_inside_plume, remove_center_flag):
    all_neighboring_points = []
    for point in top_points:
        i, j = point
        half_size = square_size // 2
        neighboring_points = [(i + di, j + dj) for di in range(-half_size, half_size + 1) for dj in range(-half_size, half_size + 1)]
        if remove_center_flag == 1:
            neighboring_points.remove(point)
        all_neighboring_points.extend(neighboring_points)

    all_neighboring_points = list(set(all_neighboring_points))  # remove duplicates
    all_neighboring_points = [(ni, nj) for ni, nj in all_neighboring_points if (ni, nj) in point_inside_plume]

    return all_neighboring_points


def get_pairs(radius, num_pts, rdn, mf, orig_points_inside_plume, ind, combined_mask, background_mask, dist_opt):
    ###########################################################
    extreme_pts_flag = 0  # 1: remove pixels with highest MF values
    dilate_flag = 1  # 1: dilate around seed pixels
    pos_flag = 0  # 1: exclude target pixels with non-positive MF values
    remove_center_flag = 0  # 1: remove seed pixels
    use_all_pts_flag = 0  # 1: use all pixels inside plume as target pixels
    allow_repetition = 0  # 1: one background spectrum can be paired with multiple target spectra; 0: unique background spectrum for each target spectrum
    ###########################################################

    background_mask = background_mask.copy()

    points_inside_plume = orig_points_inside_plume

    points_inside_plume = [(int(x), int(y)) for x, y in points_inside_plume]  # convert all points to integer tuples
    points_inside_plume = [point for point in points_inside_plume if not combined_mask[point[1], point[0]]]  # exclude bad pixels

    if extreme_pts_flag == 1:  # exclude points with MF values greater than or equal to the 99th percentile
        mf_values = [mf[point[1], point[0]] for point in points_inside_plume]
        mf_highest = np.percentile(mf_values, 99)
        print("high", mf_highest)
        points_inside_plume = [point for point in points_inside_plume if mf[point[1], point[0]] < mf_highest]

    # select the top num_pts points with the highest MF values 
    sorted_points = sorted(points_inside_plume, key=lambda point: mf[point[1], point[0]], reverse=True)
    num_pts = min(num_pts, len(points_inside_plume))
    if use_all_pts_flag == 1:
        num_pts = len(points_inside_plume)
    top_points = sorted_points[:num_pts]

    if dilate_flag == 1:  # dilate pixels
        square_size = 3
        top_points = get_neighboring_points(top_points, square_size, points_inside_plume, remove_center_flag)

    if pos_flag == 1:
        top_points = [point for point in top_points if mf[point[1], point[0]] > 0]

    top_points = [(y, x) for x, y in top_points]  # swap x y to so that it is (row, col)

    if not top_points:
        return None

    # define bounds for the region to search for background pixels
    x_coords, y_coords = zip(*top_points)
    x_coords, y_coords = np.array(x_coords), np.array(y_coords)
    min_x = max(min(x_coords) - radius, 0)
    max_x = min(max(x_coords) + radius, rdn.shape[0] - 1)
    min_y = max(min(y_coords) - radius, 0)
    max_y = min(max(y_coords) + radius, rdn.shape[1] - 1)

    A_data = rdn[x_coords, y_coords, :]
    A_data = A_data[:, ind]  # target spectra outside methane absorption region

    B_data_full = rdn[min_x:max_x + 1, min_y:max_y + 1, :]
    B_data_full = B_data_full[:, :, ind]  # candidate background spectra outside methane absorption region

    background_mask[x_coords, y_coords] = True  # mask out target pixels from background pixels
    condition_mask = background_mask[min_x:max_x + 1, min_y:max_y + 1]
    valid_indices = np.where(~condition_mask)  # select valid background pixels
    B_data = B_data_full[valid_indices[0], valid_indices[1], :]

    if B_data.shape[0] < A_data.shape[0]:
        # not enough background points to match with (reject as edge case)
        return None

    # each row of similarity_matrix corresponds to a target pixel; each column corresponds to a background pixel
    if dist_opt == 0:
        similarity_matrix = cdist(A_data, B_data, 'euclidean')  # smaller value means higher similarity
    elif dist_opt == 1:
        A_data_normalized = A_data / np.sum(np.abs(A_data), axis=1, keepdims=True)
        B_data_normalized = B_data / np.sum(np.abs(B_data), axis=1, keepdims=True)  # L1 normalized
        similarity_matrix = cdist(A_data_normalized, B_data_normalized, metric='euclidean')
    elif dist_opt == 2:
        A_norm = A_data / np.linalg.norm(A_data, axis=1, keepdims=True)
        B_norm = B_data / np.linalg.norm(B_data, axis=1, keepdims=True)  # spectral angle
        similarity_matrix = np.arccos(np.clip(np.dot(A_norm, B_norm.T), -1.0, 1.0))  # smaller value means higher similarity
    elif dist_opt == 3:
        A_norm = A_data / np.linalg.norm(A_data, axis=1, keepdims=True)
        B_norm = B_data / np.linalg.norm(B_data, axis=1, keepdims=True)
        similarity_matrix = 1 - np.dot(A_norm, B_norm.T)  # smaller value means higher similarity

    if np.any(np.isnan(similarity_matrix)):
        # special case, nans protection
        print("NaNs detected in similarity_matrix!!! [Fix] Setting the distance for them to 0...")
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0)

    if allow_repetition == 0:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(similarity_matrix)  # find the optimal pairs that minimize global cost (highest similarity)
    else:
        row_ind = np.arange(similarity_matrix.shape[0])  # row indices
        col_ind = np.argmin(similarity_matrix, axis=1)  # index of the smallest element in each row

    original_B_indices = [(valid_indices[0][i] + min_x, valid_indices[1][i] + min_y) for i in col_ind]  # convert back to original index

    # each element in top_pairs has three components: (x, y) of target pixel, (x ,y) of background pixel, similarity score
    top_pairs = [[top_points[i], original_B_indices[i], similarity_matrix[i, col_ind[i]]] for i in row_ind]

    top_pairs.sort(key=lambda x: x[2])

    num_pairs_to_select = int(0.5 * len(top_pairs))
    top_pairs = top_pairs[:num_pairs_to_select]  # choose smallest 50%

    return top_pairs


def get_ratio(rdn, top_pairs):
    top_rdns = [rdn[i, j, :] for (i, j), _, _, in top_pairs]  # radiance of target pixels
    avg_top_rdn = np.mean(top_rdns, axis=0)

    low_rdns = [rdn[i, j, :] for _, (i, j), _, in top_pairs]  # radiance of background pixels
    avg_low_rdn = np.mean(low_rdns, axis=0)

    ratio = avg_top_rdn / avg_low_rdn  # target-to-background radiance ratio
    return ratio


def find_uniform_indices(matrix):
    # check if an array has two or fewer unique values
    def has_two_or_fewer_unique_values(arr):
        return len(np.unique(arr)) <= 2

    uniform_rows = []
    for i in range(matrix.shape[0]):
        if has_two_or_fewer_unique_values(matrix[i, :]):
            uniform_rows.append(i)

    uniform_columns = []
    for j in range(matrix.shape[1]):
        if has_two_or_fewer_unique_values(matrix[:, j]):
            uniform_columns.append(j)

    return uniform_rows, uniform_columns


def calculate_magnitude(ratio, mag_opt):
    if mag_opt == 0:
        mag = 1
    elif mag_opt == 1:
        mag = np.mean(np.abs(ratio - np.mean(ratio)))  # mean absolute deviation from the mean of the ratio.
    elif mag_opt == 2:
        mag = np.linalg.norm(ratio)
    elif mag_opt == 3:
        mag = np.mean(np.abs(ratio))
    elif mag_opt == 4:
        mag = max(ratio) - min(ratio)
    elif mag_opt == 5:
        mag = np.std(ratio)
    return mag


def calculate_dist(ratio, sig, dist_opt):
    if dist_opt == 0:
        dist = np.mean(np.abs(ratio - sig))
    if dist_opt == 1:
        if np.std(ratio) == 0 or np.std(sig) == 0:
            dist = -1  # avoid division by zero
        correlation_coefficient = np.corrcoef(ratio, sig)[0, 1]
        dist = 1 - correlation_coefficient ** 2
    return dist
