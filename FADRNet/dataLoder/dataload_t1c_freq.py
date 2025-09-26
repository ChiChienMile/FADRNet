import os
import numpy as np
import nibabel as nib


def load_nii_affine(filename):
    """
    Load a NIfTI image and its affine matrix.

    Args:
        filename (str): Path to the NIfTI file (.nii)

    Returns:
        data (ndarray): 3D image array
        affine (ndarray): 4x4 affine matrix
    """
    if not os.path.exists(filename):
        return np.array([1])
    nii = nib.load(filename)
    data = nii.get_data()
    affine = nii.affine
    nii.uncache()
    return data, affine


def Padding3d(data, padding_shape=(160, 192, 160)):
    """
    Zero-pad a 3D volume to the target shape, centered.

    Args:
        data (ndarray): Input 3D volume
        padding_shape (tuple): Target shape (H, W, D)

    Returns:
        ndarray: Zero-padded 3D volume
    """
    padding_h, padding_w, padding_d = (np.array(padding_shape) - np.array(np.shape(data))) // 2

    padding_T = np.zeros([padding_h, data.shape[1], data.shape[2]], dtype=np.float32)
    padding_B = np.zeros([padding_shape[0] - padding_h - data.shape[0], data.shape[1], data.shape[2]], dtype=np.float32)
    H_data = np.concatenate([padding_T, data, padding_B], 0)

    padding_L = np.zeros([H_data.shape[0], padding_w, H_data.shape[2]], dtype=np.float32)
    padding_R = np.zeros([H_data.shape[0], padding_shape[1] - padding_w - H_data.shape[1], H_data.shape[2]], dtype=np.float32)
    W_data = np.concatenate([padding_L, H_data, padding_R], 1)

    padding_F = np.zeros([W_data.shape[0], W_data.shape[1], padding_d], dtype=np.float32)
    padding_Ba = np.zeros([W_data.shape[0], W_data.shape[1], padding_shape[2] - padding_d - W_data.shape[2]], dtype=np.float32)
    data = np.concatenate([padding_F, W_data, padding_Ba], 2)
    return data


def get_rectangle_3d(mask):
    """
    Compute the bounding box of a 3D binary mask.

    Args:
        mask (ndarray): 3D binary mask (0/1)

    Returns:
        tuple: (min_x, max_x, min_y, max_y, min_z, max_z)
    """
    mask_size = mask.shape
    min_x, min_y, min_z = mask_size[0], mask_size[1], mask_size[2]
    max_x, max_y, max_z = 0, 0, 0

    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            for k in range(mask_size[2]):
                if mask[i][j][k] > 0:
                    min_x, min_y, min_z = min(min_x, i), min(min_y, j), min(min_z, k)
                    max_x, max_y, max_z = max(max_x, i), max(max_y, j), max(max_z, k)
    return min_x, max_x, min_y, max_y, min_z, max_z


def random_crop(img_T1C, mask_input_img_T1C, mask, force_fg=True):
    """
    Perform random 3D cropping. If force_fg=True, the crop must include the tumor region.

    Args:
        img_T1C (ndarray): T1C image
        mask_input_img_T1C (ndarray): background mask of T1C
        mask (ndarray): tumor segmentation mask
        force_fg (bool): whether to force tumor region inside crop

    Returns:
        tuple: (cropped image, cropped mask_input, cropped segmentation)
    """
    P = np.random.random()
    if P < 0.8:
        crop_mask = np.zeros((160, 192, 160))
        while np.sum(crop_mask) != np.sum(mask):
            shape_im = img_T1C.shape
            crop_size = shape_im

            lb_x, ub_x = 0, shape_im[0] - crop_size[0] // 2
            lb_y, ub_y = 0, shape_im[1] - crop_size[1] // 2
            lb_z, ub_z = 0, shape_im[2] - crop_size[2] // 2

            if force_fg:
                P_minx, P_max, P_miny, P_maxy, P_minz, P_maxz = get_rectangle_3d(mask)
                # Ensure the randomly selected crop includes the tumor bounding box
                bbox_x_lb = np.random.randint(0, P_minx + 1)
                bbox_y_lb = np.random.randint(0, P_miny + 1)
                bbox_z_lb = np.random.randint(0, P_minz + 1)
            else:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + crop_size[0]
            bbox_y_ub = bbox_y_lb + crop_size[1]
            bbox_z_ub = bbox_z_lb + crop_size[2]

            # Clip to valid region inside the image
            valid_bbox_x_lb, valid_bbox_x_ub = max(0, bbox_x_lb), min(shape_im[0], bbox_x_ub)
            valid_bbox_y_lb, valid_bbox_y_ub = max(0, bbox_y_lb), min(shape_im[1], bbox_y_ub)
            valid_bbox_z_lb, valid_bbox_z_ub = max(0, bbox_z_lb), min(shape_im[2], bbox_z_ub)

            crop_img_T1C = np.copy(img_T1C[valid_bbox_x_lb:valid_bbox_x_ub,
                                           valid_bbox_y_lb:valid_bbox_y_ub,
                                           valid_bbox_z_lb:valid_bbox_z_ub])

            mask_input_img_T1C = np.copy(mask_input_img_T1C[valid_bbox_x_lb:valid_bbox_x_ub,
                                                            valid_bbox_y_lb:valid_bbox_y_ub,
                                                            valid_bbox_z_lb:valid_bbox_z_ub])

            crop_mask = np.copy(mask[valid_bbox_x_lb:valid_bbox_x_ub,
                                     valid_bbox_y_lb:valid_bbox_y_ub,
                                     valid_bbox_z_lb:valid_bbox_z_ub])

            crop_img_T1C = Padding3d(crop_img_T1C, shape_im)
            mask_input_img_T1C = Padding3d(mask_input_img_T1C, shape_im)
            crop_mask = Padding3d(crop_mask, shape_im)
            return crop_img_T1C, mask_input_img_T1C, crop_mask
    else:
        return img_T1C, mask_input_img_T1C, mask


def load_img(path, type='T1C'):
    """
    Load MRI image (no normalization here).

    Args:
        path (str): Path to dataset folder
        type (str): Image modality, default 'T1C'

    Returns:
        img (ndarray): Raw image (float)
        img_mask (ndarray): Binary mask of non-zero voxels (float)
        affine (ndarray): Affine matrix
    """
    img, affine = load_nii_affine(path + '/' + type + '.nii')
    img = np.array(img, dtype=float)
    img_mask = (img > 0).astype(float)
    return img, img_mask, affine


def _compute_mask_stats(img, mask_bool):
    """
    Compute mean and std on voxels indicated by mask_bool.

    Args:
        img (ndarray): Image volume
        mask_bool (ndarray): Boolean mask

    Returns:
        (mean, std): float, float
    """
    vox = img[mask_bool]
    # Add small epsilon to std for numerical stability
    mean = float(vox.mean()) if vox.size > 0 else 0.0
    std = float(vox.std()) if vox.size > 0 else 1.0
    return mean, (std + 1e-8)


def _normalize_with_stats(img, mask_bool, mean, std):
    """
    Normalize img in-place on mask_bool using provided mean/std.

    Args:
        img (ndarray): Image volume
        mask_bool (ndarray): Boolean mask
        mean (float): Mean
        std (float): Std

    Returns:
        ndarray: Normalized image
    """
    out = img.astype(float).copy()
    out[mask_bool] = (out[mask_bool] - mean) / std
    return out


def get_transimg(img_T1C_in, mask_input_img_T1C_in, mask_seg_in):
    """
    Augmentation pipeline for spatial domain:
    1) Random crop (optionally ensuring tumor region is included)
    2) Build background mask on the cropped image
    3) Compute mean/std on the same cropped mask
    4) Normalize the cropped image using these stats

    Returns:
        img_norm (ndarray): Cropped & normalized image
        mask_input_img (ndarray): Cropped background mask (float)
        mean (float): Mean used for normalization
        std (float): Std used for normalization
    """
    img_T1C, mask_input_img_T1C, mask_seg = random_crop(img_T1C_in, mask_input_img_T1C_in, mask_seg_in)
    mask_input_img = (mask_input_img_T1C > 0).astype(float)
    mask_bool = mask_input_img.astype(bool)

    return img_T1C, mask_bool


def load_mask(path):
    """
    Load tumor segmentation mask.

    Args:
        path (str): Dataset folder path

    Returns:
        mask (ndarray): Tumor segmentation (float array)
    """
    mask_img_path = path + '/mask.nii'
    mask, _ = load_nii_affine(mask_img_path)
    mask = np.array(mask, dtype=float)
    return mask


# This frequency mask is prepared outside (e.g., in PyTorch dataset).
# When diameter == 999, we return the original image (still normalized with the SAME stats).
def apply_frequency_mask(image_cropped, diameters_list, freqmask_list, mask_bool, mean, std):
    """
    Apply frequency masks to a cropped & normalized image.
    All frequency outputs are re-normalized using the SAME (mean, std) computed on the same crop/mask,
    ensuring consistent scaling across bands.

    Args:
        image_cropped (ndarray): Cropped image (spatial domain)
        diameters_list (list[int]): List of frequency diameters; 999 means identity (no filtering)
        freqmask_list (list[ndarray]): Corresponding frequency masks in k-space
        mask_bool (ndarray): Background mask (bool) for normalization region
        mean (float): Mean computed on the cropped image/mask
        std (float): Std computed on the cropped image/mask

    Returns:
        ndarray: Stack of frequency-filtered images, each normalized using the same stats
    """
    # Reconstruct k-space from the (already normalized) spatial image
    img_for_fft = image_cropped
    fft_image = np.fft.fftn(img_for_fft)
    fft_image_shifted = np.fft.fftshift(fft_image)

    outputs = []

    for i, d in enumerate(diameters_list):
        if d == 999:
            # Return the spatial image itself
            out = _normalize_with_stats(image_cropped, mask_bool, mean, std)
        else:
            filtered_fft_shifted = fft_image_shifted * freqmask_list[i]
            filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
            filtered_image = np.fft.ifftn(filtered_fft)
            # Magnitude image (real-valued)
            filtered_image_real = np.abs(filtered_image)

            # Normalize with the SAME stats as the base crop
            out = _normalize_with_stats(filtered_image_real, mask_bool, mean, std)

        outputs.append(out)

    return np.array(outputs)


# ==============================
# Dataloader Interfaces
# ==============================
def default_loader_freq(path, diameters_list, freqmask_list):
    """
    Training/validation loader (frequency) with augmentation.
    Crop first, compute (mean, std) on the cropped mask, and normalize ALL bands with the SAME stats.

    Args:
        path (str): Dataset folder path
        diameters_list (list[int]): Frequency band identifiers (999 = identity)
        freqmask_list (list[ndarray]): Frequency masks (k-space)

    Returns:
        ndarray: [num_bands, H, W, D] frequency-filtered & consistently normalized volumes
    """
    img_T1C, mask_input_img_T1C, _ = load_img(path, 'T1C')
    mask_seg = (load_mask(path) > 0).astype(float)

    mask_bool_before_crop = (mask_input_img_T1C > 0)
    # Compute stats on the same (test) mask and normalize the base image
    mean, std = _compute_mask_stats(img_T1C, mask_bool_before_crop)

    # Crop -> mask -> compute stats -> normalize base crop
    img_cropped, mask_bool_after_crop = get_transimg(img_T1C, mask_input_img_T1C, mask_seg)

    # Apply frequency filters and normalize each with the SAME stats
    img_freq = apply_frequency_mask(img_cropped, diameters_list, freqmask_list, mask_bool_after_crop, mean, std)
    return img_freq


def default_loader_test_freq(path, diameters_list, freqmask_list):
    """
    Test loader (frequency) without augmentation.
    Compute (mean, std) on the test mask once and reuse them for all frequency bands.

    Args:
        path (str): Dataset folder path
        diameters_list (list[int]): Frequency band identifiers (999 = identity)
        freqmask_list (list[ndarray]): Frequency masks (k-space)

    Returns:
        ndarray: [num_bands, H, W, D] frequency-filtered & consistently normalized volumes
    """
    img_T1C, mask_input_img_T1C, _ = load_img(path, 'T1C')
    mask_bool = (mask_input_img_T1C > 0)

    # Compute stats on the same (test) mask and normalize the base image
    mean, std = _compute_mask_stats(img_T1C, mask_bool)
    # Use the SAME stats for all bands
    img_freq = apply_frequency_mask(img_T1C, diameters_list, freqmask_list, mask_bool, mean, std)
    return img_freq
