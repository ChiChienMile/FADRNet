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
    Load and normalize MRI image.

    Args:
        path (str): Path to dataset folder
        type (str): Image modality, default 'T1C'

    Returns:
        img (ndarray): Normalized image
        img_mask (ndarray): Binary mask of non-zero voxels
        affine (ndarray): Affine matrix
    """
    img, affine = load_nii_affine(path + '/' + type + '.nii')
    img = np.array(img, dtype=float)
    img_mask = (img > 0)
    smooth = 1e-8
    img[img_mask] = (img[img_mask] - img[img_mask].mean()) / (img[img_mask].std() + smooth)
    img_mask = np.array(img_mask, dtype=float)
    return img, img_mask, affine


def get_transimg(img_T1C_in, mask_input_img_T1C_in, mask_seg_in):
    """
    Data augmentation: apply random crop and keep the background region.

    Args:
        img_T1C_in (ndarray): Input T1C image
        mask_input_img_T1C_in (ndarray): Input mask of T1C
        mask_seg_in (ndarray): Segmentation mask

    Returns:
        ndarray: Cropped and masked image
    """
    img_T1C, mask_input_img_T1C, mask_seg = random_crop(img_T1C_in, mask_input_img_T1C_in, mask_seg_in)
    mask_input_img = np.array((mask_input_img_T1C) > 0, dtype=float)
    img_original = img_T1C * mask_input_img
    return img_original


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


# ==============================
# Dataloader Interfaces
# ==============================

def default_loader(path):
    """
    Training/validation loader with augmentation.

    Args:
        path (str): Dataset folder path

    Returns:
        ndarray: Preprocessed training image
    """
    img_T1C, mask_input_img_T1C, affine = load_img(path, 'T1C')
    mask_seg = load_mask(path)
    mask_seg = (mask_seg > 0).astype(float)
    img_original = get_transimg(img_T1C, mask_input_img_T1C, mask_seg)
    return img_original


def default_loader_test(path):
    """
    Test loader without augmentation.

    Args:
        path (str): Dataset folder path

    Returns:
        ndarray: Preprocessed test image
    """
    img_T1C, mask_input_img_T1C, affine = load_img(path, 'T1C')
    mask_input_img = np.array((mask_input_img_T1C) > 0, dtype=float)
    mask_original = img_T1C * mask_input_img
    return mask_original