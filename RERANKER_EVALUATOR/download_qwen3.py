import os
from huggingface_hub import snapshot_download

# 1. Cấu hình đường dẫn
# Thay đổi đường dẫn này nếu bạn muốn lưu ở chỗ khác
model_id = "Qwen/Qwen3-VL-Reranker-8B"
local_dir = "MODELS/Qwen3-VL-Reranker-8B"

# Tạo thư mục nếu chưa có
os.makedirs(local_dir, exist_ok=True)

print(f"--- Đang bắt đầu tải model: {model_id} ---")
print(f"--- Đích đến: {local_dir} ---")

try:
    # Tải toàn bộ repository
    # local_dir_use_symlinks=False: Rất quan trọng cho Slurm Cluster để file thật được lưu 
    # thay vì các đường dẫn ảo (symlinks), tránh lỗi khi các node khác nhau truy cập.
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main", # Hoặc bản cụ thể nếu cần
        resume_download=True, # Cho phép tải tiếp nếu bị ngắt mạng
        max_workers=8 # Tăng tốc độ tải
    )
    print("\n--- TẢI MODEL THÀNH CÔNG! ---")
    
    # Kiểm tra xem các file quan trọng đã có chưa
    files = os.listdir(local_dir)
    print(f"Danh sách file đã tải: {files}")
    
except Exception as e:
    print(f"\nCó lỗi xảy ra trong quá trình tải: {e}")