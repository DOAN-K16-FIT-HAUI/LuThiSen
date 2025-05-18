from __future__ import annotations

import json
import unittest

import cv2
import requests  # type: ignore


class TestTextOCRAPI(unittest.TestCase):

    def setUp(self) -> None:
        """Cài đặt ban đầu"""
        self.image_path = r'E:\DATN\DATN_LuThiSen\src\model_deployed\processed_output.jpg'
        self.api_url = 'http://localhost:5000/v1/text_ocr'  # Thay đổi nếu API khác

        # Bboxes, classes và confs từ kết quả thực tế
        self.bboxes = [
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
        self.classes = [
            'Course', 'Name', 'Msv', 'HKTT', 'Date', 'Class',
        ]

    def test_text_ocr_with_real_data(self):
        """Test Text OCR API với dữ liệu bounding box thực tế"""
        img = cv2.imread(self.image_path).tolist()

        payload = {
            'img': img,
            'bboxes': self.bboxes,
            'classes': self.classes,
        }

        response = requests.post(self.api_url, json=payload)

        print(f'Status Code: {response.status_code}')
        print('Raw response:', response.text)

        self.assertEqual(response.status_code, 200)

        response_json = response.json()

        self.assertIn('info', response_json)
        info = response_json['info']
        print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
