"""
Các tiện ích chuẩn bị dữ liệu trước khi đi vào bước embedding.

Mục tiêu của package này:
- không embed trực tiếp
- không gọi API model
- chỉ biến dữ liệu đã extract thành artifact sạch, ổn định, dễ kiểm tra

Artifact ở đây là lớp trung gian giữa:
`data/processed_data/` -> `data/embedding_data/`
"""

