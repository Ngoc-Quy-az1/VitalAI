# Hướng dẫn Triển khai Dịch vụ Medical Tools lên AWS

Tài liệu này hướng dẫn chi tiết cách build Docker image, chạy thử nghiệm local bằng Docker Compose, và triển khai lên đám mây **Amazon Web Services (AWS)** bằng 2 phương thức phổ biến và tối ưu nhất: **AWS App Runner** (Khuyên dùng - Nhanh, rẻ, dễ quản lý) và **AWS ECS Fargate** (Chuyên nghiệp cho doanh nghiệp).

---

## 1. Chạy thử nghiệm Local (Docker & Docker Compose)

Đảm bảo bạn đang ở thư mục gốc của dự án (`VitalAI/`):

### A. Build Docker Image thủ công
```bash
docker build -f docker/Dockerfile.medical_tools -t vitalai-medical-tools:latest .
```

### B. Khởi động bằng Docker Compose (Khuyên dùng để test)
Di chuyển vào thư mục `docker/` và chạy:
```bash
cd docker
docker-compose up --build -d
```
* API sẽ lắng nghe tại cổng `http://localhost:8010`.
* Truy cập `http://localhost:8010/health` để kiểm tra trạng thái hoạt động.
* Xem tài liệu API tự động tại `http://localhost:8010/docs`.

---

## 2. Đẩy Docker Image lên AWS ECR (Elastic Container Registry)

AWS ECR là nơi lưu trữ Docker image riêng tư của bạn trước khi deploy.

### Bước 2.1: Đăng nhập vào AWS CLI
*(Đảm bảo đã cấu hình AWS CLI bằng lệnh `aws configure`)*
```bash
aws ecr get-login-password --region <AWS_REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
```

### Bước 2.2: Tạo Repo mới trên ECR (nếu chưa có)
```bash
aws ecr create-repository --repository-name vitalai-medical-tools --region <AWS_REGION>
```

### Bước 2.3: Tag và Push Image lên ECR
```bash
# Tag image local sang định dạng ECR
docker tag vitalai-medical-tools:latest <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/vitalai-medical-tools:latest

# Đẩy image lên ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/vitalai-medical-tools:latest
```

---

## 3. Lựa chọn Triển khai trên AWS

> [!TIP]
> **Khuyên dùng AWS App Runner:** Rất thích hợp cho các dịch vụ API viết bằng FastAPI/Uvicorn. Nó tự động quản lý server, SSL, tự động scale (Auto-scaling), chi phí rất rẻ (chỉ tính tiền khi có request) và cực kỳ dễ thiết lập.

### Lựa chọn A: AWS App Runner (Đơn giản nhất & Tối ưu nhất)
1. Truy cập **AWS App Runner Console**.
2. Chọn **Create service**.
3. **Source**: Chọn **Container registry** -> Chọn **Amazon ECR**.
4. **Container image URI**: Trỏ tới Repo ECR của bạn (chọn tag `latest`).
5. **Deployment settings**: Chọn **Automatic** (App Runner sẽ tự động cập nhật web service mỗi khi bạn push bản build mới lên ECR!).
6. **Service settings**:
   - **Port**: Điền `8010`.
   - **Environment variables** (Biến môi trường): Thêm các khóa từ file `.env` của bạn như:
     - `MISTRAL_CLIENT_API_KEY`: Key dịch vụ Mistral OCR.
     - `OPENAI_API_KEY`: Key OpenAI làm phương án dự phòng (Fallback).
7. Nhấn **Create & Deploy**. Sau 3-5 phút, bạn sẽ nhận được một địa chỉ URL HTTPS công khai (ví dụ: `https://xxxx.awsapprunner.com`) để frontend kết nối trực tiếp!

---

### Lựa chọn B: AWS ECS Fargate (Serverless Container cho dự án lớn)
Nếu dự án của bạn cần tích hợp sâu vào hệ thống VPC mạng nội bộ của AWS:
1. **Tạo ECS Cluster**: Vào AWS ECS, chọn **Create Cluster**, chọn mẫu **Fargate (Serverless)**.
2. **Tạo Task Definition**:
   - Chọn loại khởi chạy **Fargate**.
   - Cấu hình Task Size: `0.5 vCPU` và `1 GB RAM` (Dịch vụ này siêu nhẹ nên không cần cấu hình quá cao).
   - Thêm Container: Cấu hình Container Name, nhập **Image URI** từ ECR.
   - Map Port: Host Port `8010` -> Container Port `8010`.
   - Cấu hình Environment Variables (Biến môi trường) như ở mục trên.
3. **Tạo Service**:
   - Chạy Service trong ECS Cluster của bạn.
   - Cấu hình Network: Chọn các Subnet và tạo Security Group cho phép truy cập cổng `8010`.
   - Kết nối với **Application Load Balancer (ALB)** để cấp URL public và cấu hình SSL HTTPS.

---

## 4. Các biến môi trường cần lưu ý trên Cloud
Khi deploy, hãy đảm bảo bạn cấu hình đầy đủ các biến môi trường sau trong phần thiết lập Container:

| Tên biến | Kiểu giá trị | Mô tả |
| :--- | :--- | :--- |
| `MEDICAL_TOOLS_DATA_DIR` | `/app/data/processed_data` | Giữ nguyên đường dẫn mặc định trong container |
| `MISTRAL_CLIENT_API_KEY` | `xxxx-xxxx` | API Key của Mistral để chạy OCR chính |
| `OPENAI_API_KEY` | `sk-proj-xxxx` | API Key của OpenAI để chạy OCR fallback khi Mistral lỗi |
