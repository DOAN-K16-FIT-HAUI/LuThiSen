from __future__ import annotations

from functools import cached_property
from typing import List

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from ultralytics import YOLO

logger = get_logger(__name__)


class TextDetectorModelInput(BaseModel):
    img_processed: np.ndarray


class TextDetectorModelOutput(BaseModel):
    class_list: List[str]
    bboxes_list: List[np.ndarray]
    conf_list: List[float]


class TextDetectorModel(BaseService):
    settings: Settings

    @cached_property
    def model_loaded(self):
        # Load the face detection model here
        return YOLO(self.settings.text_detector.model_path)

    async def process(self, inputs: TextDetectorModelInput) -> TextDetectorModelOutput:

        # Perform face detection using the model
        class_list, bboxes_list, scores_list = self.forward(
            inputs.img_processed, self.settings.text_detector.conf,
        )

        return TextDetectorModelOutput(
            bboxes_list=bboxes_list,
            class_list=class_list,
            conf_list=scores_list,
        )

    def forward(self, img: np.ndarray, threshold: float) -> tuple:
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
        scores_list = []
        bboxes_list = []
        class_list = []

        model = self.model_loaded
        results = model(img)  # Lấy kết quả đầu tiên
        for result in results:
            for box in result.boxes:
                # Lấy confidence score
                # Lấy giá trị score từ Tensor thành float
                score = float(box.conf[0].item())

                # Lấy tọa độ bounding box
                # Chuyển Tensor về NumPy array
                bbox = box.xyxy[0].cpu().numpy()

                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Thêm vào danh sách
                scores_list.append(score)
                bboxes_list.append(bbox)
                class_list.append(class_name)
        # Sort theo score rồi sd nms để lọc ra các bbox có score lớn nhất(vì bài toán chỉ nhận vào
        # 1 thẻ sinh viên / 1 ảnh

        return class_list, bboxes_list, scores_list

    def nms(self, dets):
        thresh = self.settings.detector.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
