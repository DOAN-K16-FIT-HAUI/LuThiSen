from __future__ import annotations

from functools import cached_property
from typing import Tuple

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from ultralytics import YOLO

logger = get_logger(__name__)


class CardDetectorModelInput(BaseModel):
    img: np.ndarray


class CardDetectorModelOutput(BaseModel):
    bboxes: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)


class CardDetectorModel(BaseService):
    settings: Settings

    @cached_property
    def model_loaded(self) -> YOLO:
        return YOLO(self.settings.card_detector.model_path)

    async def process(self, inputs: CardDetectorModelInput) -> CardDetectorModelOutput:
        scores, bboxes = self.forward(
            inputs.img, self.settings.card_detector.conf,
        )
        return CardDetectorModelOutput(bboxes=bboxes, scores=scores)

    def forward(self, img: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass on the card detection model to extract all bounding boxes and confidence scores,
        then applies NMS and sorts them in descending order of score.
        """
        model = self.model_loaded
        results = model(img)[0]

        det_list = []
        for box in results.boxes:
            score = box.conf[0].item()
            if score < threshold:
                continue
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            det_list.append(np.append(bbox, score))

        if not det_list:
            return np.empty((0,), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        detections_array = np.array(det_list, dtype=np.float32).reshape(-1, 5)

        # Apply NMS
        keep_indices = self.nms(detections_array)
        detections_nms = detections_array[keep_indices]

        # Sort by score descending
        sorted_indices = np.argsort(detections_nms[:, 4])[::-1]
        sorted_detections = detections_nms[sorted_indices]

        scores = sorted_detections[:, 4]
        bboxes = sorted_detections[:, :4]

        return scores, bboxes

    def nms(self, dets: np.ndarray) -> list[int]:
        thresh = self.settings.card_detector.conf
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
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
