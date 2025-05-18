from __future__ import annotations

import asyncio
import unittest

import cv2
from common import get_settings
from infrastructure.text_ocr import TextOCRModel
from infrastructure.text_ocr import TextOCRModelInput


class TestTextOCROnly(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_ocr = TextOCRModel(settings=self.settings)

    def test_text_ocr(self):
        # Đường dẫn ảnh cần test
        image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/text_detector_preproces.png'
        img = cv2.imread(image_path)
        self.assertIsNotNone(img, 'Ảnh không được load thành công!')

        # Dữ liệu Input từ bạn cung cấp
        class_list = [
            'Course', 'Name', 'Msv', 'HKTT', 'Date', 'Class',
        ]

        bboxes_list = [
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

        # Chuyển đổi về định dạng phù hợp (float -> int)
        bboxes_list = [list(map(int, box)) for box in bboxes_list]

        # Tạo input cho model OCR
        text_input = TextOCRModelInput(
            img=img,
            class_list=class_list,
            bboxes_list=bboxes_list,
        )

        # Gọi hàm xử lý OCR
        text_output = asyncio.run(self.text_ocr.process(text_input))

        # In kết quả ra console
        for result in text_output.results:
            print(result)
        print(text_output.results)
        # Vẽ kết quả lên ảnh
        # img_output = img.copy()
        # for result in text_output.results:
        #     x1, y1, x2, y2 = map(int, result['bounding_box'])
        #     cls = result['class']
        #     text = result['text']

        #     cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(
        #         img_output,
        #         f'{cls}: {text}',
        #         (x1, max(y1 - 10, 0)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        # # Lưu ảnh có kết quả OCR
        # cv2.imwrite('text_ocr_output.png', img_output)


if __name__ == '__main__':
    unittest.main()
