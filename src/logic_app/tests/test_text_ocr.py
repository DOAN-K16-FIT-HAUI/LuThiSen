from __future__ import annotations

import json
import unittest

import cv2
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder
from infrastructure.text_ocr import TextOCR
from infrastructure.text_ocr import TextOCRInput


class TestTextOCR(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_ocr = TextOCR(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'

        self.bboxes_list = [
            [
                634.2853393554688,
                589.3724365234375,
                890.7731323242188,
                639.2056884765625,
            ],
            [
                525.900146484375,
                301.1031799316406,
                796.80859375,
                361.1376647949219,
            ],
            [
                51.94629669189453,
                692.9230346679688,
                273.922119140625,
                730.7155151367188,
            ],
            [
                502.302978515625,
                444.47406005859375,
                1146.979736328125,
                502.43115234375,
            ],
            [
                650.5380859375,
                375.4815673828125,
                921.237548828125,
                422.485595703125,
            ],
            [
                453.7305908203125,
                517.524169921875,
                1101.737548828125,
                569.0948486328125,
            ],
        ]

        self.class_list = [
            'Course', 'Name', 'Msv', 'HKTT', 'Date', 'Class',
        ]

    def test_api_text_ocr(self):
        test_image = cv2.imread(self.img_path)
        if test_image is None:
            raise FileNotFoundError(f'Không thể đọc ảnh tại: {self.img_path}')
        print('Image shape:', test_image.shape, 'dtype:', test_image.dtype)

        inputs = TextOCRInput(
            img=test_image,
            class_list=self.class_list,
            bboxes_list=self.bboxes_list,
        )

        result = self.text_ocr.process(inputs=inputs)

        # ✅ In kết quả dưới dạng JSON đẹp (pretty-print)
        print('\n--- RESPONSE JSON ---')
        result_json = jsonable_encoder(result)
        print(json.dumps(result_json, indent=4, ensure_ascii=False))

        # Nếu có kết quả cụ thể hơn cần in
        print('\n--- RESULTS ---')
        for i, r in enumerate(result.results):
            print(f'[{i}] {r}')

        print('Số lượng kết quả:', len(result.results))


if __name__ == '__main__':
    unittest.main()
