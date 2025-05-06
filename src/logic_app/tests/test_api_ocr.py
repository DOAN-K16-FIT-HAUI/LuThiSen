from __future__ import annotations

import json
import os
import unittest

import requests  # type: ignore


class TestOCRAPI(unittest.TestCase):
    def setUp(self) -> None:
        # Đường dẫn tới ảnh mẫu dùng để test
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'
        self.api_url = 'http://localhost:5001/v1/ocr'

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found: {self.image_path}')

    def test_ocr(self):
        # Đọc ảnh và gửi yêu cầu POST tới API OCR
        with open(self.image_path, 'rb') as img_file:
            files = {
                'file': ('sample_image.jpg', img_file, 'image/jpeg'),
            }
            response = requests.post(self.api_url, files=files)

        print('Status Code:', response.status_code)

        try:
            json_data = response.json()

            # ✅ In đẹp JSON kết quả
            print('\n🎯 Response JSON:\n')
            print(json.dumps(json_data, indent=4, ensure_ascii=False))

            # ✅ Kiểm tra phản hồi hợp lệ
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', json_data)
            self.assertEqual(json_data['message'], 'Process successfully !!!')

        except Exception as e:
            print('\n❌ Failed to parse JSON response:', e)
            print('Raw Response:', response.text)
            self.fail('Invalid JSON response received.')


if __name__ == '__main__':
    unittest.main()
