import numpy as np
import cv2
from monai.config import KeysCollection
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.compose import MapTransform

class CLAHEd(MapTransform):
    """
    """
    def __init__(self, keys: KeysCollection, clip_limit, tile_grid_size) -> None:
        super().__init__(keys)
        self.scaler = CLAHE(clip_limit, tile_grid_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.scaler(d[key])
        return d

class CLAHE(Transform):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        clahe_mat = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = img.astype('uint8')
            img = clahe_mat.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        return img
