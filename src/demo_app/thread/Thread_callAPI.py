from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests  # type: ignore
from model.base_result import BaseInfo
from PyQt5.QtCore import QThread


class APICallerThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        self.exec_()

    def prepare_image_file(self, image: np.ndarray):
        """Convert np.ndarray image to file-like object for upload"""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError('Failed to encode image to JPEG format')

        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = 'image.jpg'
        file_bytes.seek(0)
        return file_bytes

    def call_api(self, api_url: str, data):
        try:
            if isinstance(data, np.ndarray):
                # Nếu là ảnh numpy
                file_bytes = self.prepare_image_file(data)
                files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
                response = requests.post(api_url, files=files)
            else:
                # Gửi JSON thông thường
                response = requests.post(api_url, json=data)

            response.raise_for_status()
            json_data = response.json()
            print(f'Phản hồi từ {api_url}: {json_data}')

            info_text = json_data['info']['info_text'][0]

            return BaseInfo(
                cls=info_text.get('cls', ''),
                course=info_text.get('course', ''),
                date=info_text.get('date', ''),
                hktt=info_text.get('hktt', ''),
                msv=info_text.get('msv', ''),
                name=info_text.get('name', ''),
            )

        except requests.RequestException as e:
            print(f'Lỗi khi gọi {api_url}: {e}')
        except KeyError as e:
            print(f'Lỗi trích xuất dữ liệu từ response: {e}')
        except Exception as e:
            print(f'Lỗi không xác định: {e}')

        return None
