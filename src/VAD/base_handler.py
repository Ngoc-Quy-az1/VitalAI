

class BaseHandler:
    def setup(self, **kwargs):
        """Khởi tạo trạng thái; lớp con ghi đè với tham số cụ thể."""
        raise NotImplementedError

    def process(self, *args, **kwargs):
        """Xử lý một bước/luồng; lớp con triển khai."""
        raise NotImplementedError

    def on_session_end(self):
        """Gọi khi kết phiên (reset buffer, model state)."""
        pass
