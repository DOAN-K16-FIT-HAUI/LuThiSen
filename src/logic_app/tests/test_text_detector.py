from __future__ import annotations

import json
import unittest

import cv2
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder
from infrastructure.text_detector import TextDetector
from infrastructure.text_detector import TextDetectorInput


class TestTextDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_detector = TextDetector(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'

    def test_api_text_detection(self):
        test_image = cv2.imread(self.img_path)
        print(
            '🖼️ Image shape:', test_image.shape,
            '| dtype:', test_image.dtype,
        )

        dummy_bbox = [
            341.795654296875, 896.8245849609375,
            1506.250732421875, 1659.7244873046875,
        ]
        inputs = TextDetectorInput(img_origin=test_image, bbox=dummy_bbox)

        result = self.text_detector.process(inputs=inputs)

        # ✅ In kết quả dưới dạng JSON format đẹp
        result_json = jsonable_encoder(result)
        print('\n🎯 JSON Result:\n')
        print(json.dumps(result_json, indent=4, ensure_ascii=False))

        # # ✅ Nếu muốn in riêng từng phần:
        # print('\n📦 Bboxes:', result.bboxes_list)
        # print('🏷️ Classes:', result.class_list)
        # print('📊 Confidences:', result.conf_list)


if __name__ == '__main__':
    unittest.main()
