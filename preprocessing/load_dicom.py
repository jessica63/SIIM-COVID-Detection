import os
from typing import Optional, List

import cv2
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(
    dcm_path,
    voi_lut=False,
    fix_monochrome=True,
    normalization=False,
    apply_window=False,
    range_correct=False
) -> np.ndarray:
    dicom = pydicom.read_file(dcm_path)
    # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
    elem = dicom[0x0028, 0x0101]
    elem.value = 16

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM
    # data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if range_correct:
        median = np.median(data)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.where(data == 0, median, data)
        else:
            data = np.where(data == 4095, median, data)

    if normalization:
        if apply_window and "WindowCenter" in dicom and "WindowWidth" in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)
            y_min = (window_center - 0.5 * window_width)
            y_max = (window_center + 0.5 * window_width)
        else:
            y_min = data.min()
            y_max = data.max()
        data = (data - y_min) / (y_max - y_min)
        data = np.clip(data, 0, 1)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    return data


def save_dcm_to_img(
    dcm_path: str,
    save_dir: str = "/work/VinBigData/preprocessed",
    clahe_args: Optional[List[dict]] = None,
    force_replace: bool = False,
    save_fileExt: str = "png",
    return_pixel_data: bool = False
):
    # Check save file extension
    assert save_fileExt in ["png", "jpg", "npz"], (
        f"Got unexpected value: \"{save_fileExt}\" of `save_fileExt`"
    )
    save_fname = os.path.basename(dcm_path.replace("dcm", save_fileExt))
    save_fpath = os.path.join(save_dir, save_fname)

    if not force_replace and os.path.isfile(save_fpath):
        if save_fileExt == "npz":
            data = np.load(save_fpath)["img"]
        elif save_fileExt in ["png", "jpg"]:
            data = cv2.imread(save_fpath)
    else:
        data = read_xray(
            dcm_path=dcm_path,
            voi_lut=False,
            fix_monochrome=True,
            normalization=True,
            apply_window=True,
            range_correct=True
        )
        # Convert to uint8
        data = (data * 255.).astype(np.uint8)

        # apply clahe
        if isinstance(clahe_args, list):
            concate_imgs = [data]
            # TODO: check the arguments for CLAHE method is valid
            for clahe_arg in clahe_args:
                clahe = cv2.createCLAHE(**clahe_arg)
                clahe_img = clahe.apply(data)
                concate_imgs.append(clahe_img)
            concate_imgs = np.dstack(concate_imgs).astype(np.uint8)
            data = concate_imgs

        # Save to numpy file
        if save_fileExt == "npz":
            np.savez_compressed(save_fpath, img=data)
        elif save_fileExt in ["png", "jpg"]:
            cv2.imwrite(save_fpath, data)

    shape = data.shape[:2]

    if return_pixel_data:
        if data.ndim == 3:
            # Only reture the first channel, i.e. the original image
            data = data[..., 0]
        data = data.astype(np.float32) / 255.
        return shape, data
    return shape
    # end


if __name__ == "__main__":
    save_dcm_to_img(
        dcm_path="/data2/chest_xray/siim-covid19-detection/train/00086460a852/9e8302230c91/65761e66de9f.dcm",
        save_dir="."
    )
