from __future__ import annotations

import asyncio
import json
import unittest

import cv2
import numpy as np
from common import get_settings
from infrastructure.text_detector import TextDetectorModel
from infrastructure.text_detector import TextDetectorModelInput


class TestTextDetectorOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_detector = TextDetectorModel(settings=self.settings)

    def test_text_detector(self):
        # Đường dẫn ảnh cần test
        image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'
        img = cv2.imread(image_path)
        self.assertIsNotNone(img, 'Ảnh không được load thành công!')

        # Tạo input
        text_input = TextDetectorModelInput(
            img_processed=img,
        )

        # Gọi hàm xử lý
        text_output = asyncio.run(self.text_detector.process(text_input))

        # Chuyển bboxes (numpy array) sang list để có thể serialize bằng JSON
        bboxes = [
            bbox.tolist() if isinstance(bbox, np.ndarray)
            else bbox for bbox in text_output.bboxes_list
        ]

        # Tạo dict kết quả
        result = {
            'Classes': text_output.class_list,
            'Bboxes': bboxes,
            'Confidences': text_output.conf_list,
        }

        # In ra dưới dạng JSON
        print(json.dumps(result, ensure_ascii=False, indent=4))

        cv2.imwrite(
            'text_detector_preproces.png',
            img,
        )


if __name__ == '__main__':
    unittest.main()
