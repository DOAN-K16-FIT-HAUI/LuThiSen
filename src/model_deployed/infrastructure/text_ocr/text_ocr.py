from __future__ import annotations

from functools import cached_property
from typing import List

import cv2
import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from PIL import Image
from service.card_align import CardAlignModel
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

logger = get_logger(__name__)


class TextOCRModelInput(BaseModel):
    img: np.ndarray
    class_list: List[str]
    bboxes_list: List[List[float]]


class TextOCRModelOutput(BaseModel):
    results: dict


class TextOCRModel(BaseService):
    settings: Settings

    @cached_property
    def card_align(self) -> CardAlignModel:
        return CardAlignModel(settings=self.settings)

    @cached_property
    def model_loaded(self):
        # Load the face detection model here
        config = Cfg.load_config_from_name(self.settings.text_ocr.config_name)
        # Use GPU if available
        config['device'] = self.settings.text_ocr.device
        return Predictor(config)

    async def process(self, inputs: TextOCRModelInput) -> TextOCRModelOutput:

        # processed_img = self.card_align.process2white_black(img=inputs.img)
        # cv2.imwrite('/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/cropped_outputs/processed_img2.png', processed_img)
        extracted_results = dict()

        for cls, bbox in zip(inputs.class_list, inputs.bboxes_list):
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Cắt vùng ảnh theo bounding box
            cropped_img = inputs.img[y_min:y_max, x_min:x_max]
            processed_img = self.card_align.process2white_black(
                img=cropped_img,
            )

            # Dự đoán văn bản trong vùng ảnh
            text = self.forward(processed_img)
            extracted_results[cls] = text
            # filename = f"/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/cropped_outputs/_{cls}.png"
            # cv2.imwrite(filename, cropped_img)

        return TextOCRModelOutput(
            results=extracted_results,
        )

    def forward(self, img: np.ndarray) -> tuple:
        """
        Performs a forward pass on the face detection model to extract bounding boxes, keypoints, and confidence scores.

        Args:
            img (np.ndarray): Input image in BGR format as a NumPy array.
            threshold (float): Confidence threshold for filtering detected objects.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - A list of NumPy arrays containing confidence scores for detected bounding boxes.
                - A list of NumPy arrays containing bounding box coordinates.
                - A list of NumPy arrays containing keypoint coordinates if available.
        """

        img_prepared = self.prepare_img(img)

        text_model = self.model_loaded.predict(
            img_prepared,
        )  # Lấy kết quả đầu tiên

        return text_model

    def ndarray2PIL(self, img: np.ndarray):
        if isinstance(img,  np.ndarray):
            return Image.fromarray(img)
        return None

    def prepare_img(self, img: np.ndarray):
        if isinstance(img, np.ndarray):
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        raise ValueError('Input image is not a valid numpy array.')
