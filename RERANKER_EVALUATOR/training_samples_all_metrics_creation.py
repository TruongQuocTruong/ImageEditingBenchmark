import json
import os
import random

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dùng os.path.join để nối BASE_DIR với các thư mục con
BENCHMARK_ROOT = os.path.join(BASE_DIR, "training_samples_creation", "training_samples_ver01")
EDITED_ROOT    = os.path.join(BASE_DIR, "training_samples_creation", "edited_training_samples_ver01")

# Thư mục chứa các file đã qua lọc ĐỒNG THUẬN (Consensus)
INPUT_DIR      = os.path.join(BASE_DIR, "training_samples_creation", "revise_training_samples_groundtruth")
OUTPUT_DIR     = os.path.join(BASE_DIR, "training_datasets")

# Danh sách các file metrics đã được tinh khiết hóa (Consensus)
METRIC_FILES = [
    "fidelity_training_samples.json",
    "realism_training_samples.json",
    "aesthetic_training_samples.json",
    "background_consistency_training_samples.json",
    "foreground_consistency_training_samples.json",
    "structure_consistency_training_samples.json"
]

OUTPUT_TRAIN_JSONL = os.path.join(OUTPUT_DIR, "all_metrics_train.jsonl")
OUTPUT_VAL_JSONL   = os.path.join(OUTPUT_DIR, "all_metrics_val.jsonl")

# --- 2. ĐỊNH NGHĨA SYSTEM PROMPT & METRICS (Chuẩn Reranker) ---
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be 'yes' or 'no'."
)

METRIC_DEFINITIONS = {
    "Fidelity": "Focus on how accurately the output executes the specific change requested in the instruction without adding unrequested changes.",
    "Realism": "Focus on visual plausibility, lighting consistency, shadows, perspective, and the absence of digital artifacts.",
    "Aesthetic": "Focus on artistic quality, visual harmony, composition, and color balance.",
    "Background Consistency": "Focus on the preservation of image regions unrelated to the instruction.",
    "Foreground Consistency": "Focus on object identity preservation.",
    "Structure Consistency": "Focus on the preservation of the global geometric layout and object shapes."
}

def process_all_metrics_consensus(val_ratio=0.05):
    all_raw_data = []
    
    # --- BƯỚC 1: ĐỌC VÀ GỘP TẤT CẢ DỮ LIỆU ĐỒNG THUẬN ---
    for file_name in METRIC_FILES:
        file_path = os.path.join(INPUT_DIR, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_raw_data.extend(data)
                print(f"Loaded {len(data)} consensus samples from {file_name}")
        else:
            print(f"Warning: {file_path} not found. Hãy chạy script lọc consensus trước.")

    if not all_raw_data:
        print("No data found!")
        return

    # --- BƯỚC 2: CHIA TẬP TRAIN/VAL DỰA TRÊN IMAGE_ID ---
    PRIORITY_VAL_IDS = {"000033", "000677"}

    all_image_ids = list(set(item['image_id'] for item in all_raw_data))
    remaining_ids = [i for i in all_image_ids if i not in PRIORITY_VAL_IDS]
    random.seed(42)
    random.shuffle(remaining_ids)

    num_val_ids = max(1, int(len(all_image_ids) * val_ratio))
    # Ưu tiên priority IDs vào val, lấy thêm từ remaining nếu cần
    extra_needed = max(0, num_val_ids - len(PRIORITY_VAL_IDS))
    val_ids = PRIORITY_VAL_IDS | set(remaining_ids[:extra_needed])
    
    print(f"\n--- Global Split across 6 Metrics ---")
    print(f"Total unique images: {len(all_image_ids)}")
    print(f"Images for Validation: {len(val_ids)}")

    train_samples = []
    val_samples = []

    # --- BƯỚC 3: TẠO SAMPLE THEO FORMAT PAIRWISE-TO-BINARY ---
    for item in all_raw_data:
        img_id, cat, e_type = item['image_id'], item['category'], item['edit_type']
        mod_a, mod_b = item['image_a_model'], item['image_b_model']
        instr = item['instruction']
        metric_key = item['metric']
        winner = item['selected_image']
        
        src = f"{BENCHMARK_ROOT}/{cat}/images/{img_id}.jpg"
        p_a = f"{EDITED_ROOT}/edited_{mod_a}/{cat}/{img_id}/{e_type}_{img_id}.png"
        p_b = f"{EDITED_ROOT}/edited_{mod_b}/{cat}/{img_id}/{e_type}_{img_id}.png"

        if not (os.path.exists(src) and os.path.exists(p_a) and os.path.exists(p_b)):
            continue

        definition = METRIC_DEFINITIONS.get(metric_key, "")
        
        # Format Pairwise-to-Binary: Hỏi Candidate 1 (A) có tốt hơn Candidate 2 (B) không
        user_value = (
            f"<Instruct>: Metric: {metric_key}. Definition: {definition} "
            f"Edit Instruction: {instr} "
            f"Task: Based on the Source Image in the Query, evaluate the two candidate edited images in the Document. "
            f"Does [Candidate 1] perform BETTER than [Candidate 2] in satisfying the instruction and metric?\n"
            f"<Query>:[Source Image]: <image>\n"
            f"<Document>:[Candidate 1]: <image>\n"
            f"[Candidate 2]: <image>"
        )

        a_won = (winner == "Image A")

        sample = {
            "id": f"{metric_key.lower()}_{img_id}_{e_type}_{mod_a}_{mod_b}",
            "system": SYSTEM_PROMPT,
            "conversations": [
                {"from": "human", "value": user_value}, # Chuẩn ShareGPT cho Llama-Factory
                {"from": "gpt", "value": "yes" if a_won else "no"}
            ],
            "images": [src, p_a, p_b]
        }

        if img_id in val_ids:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    # --- BƯỚC 4: XÁO TRỘN TỔNG THỂ ---
    # Việc xáo trộn giúp mô hình học đan xen các metrics, không bị học theo cụm
    random.shuffle(train_samples)

    # Xuất file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    with open(OUTPUT_VAL_JSONL, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n--- SUCCESS: ALL METRICS COMBINED ---")
    print(f"Total Training Samples: {len(train_samples)}")
    print(f"Total Validation Samples: {len(val_samples)}")
    print(f"Format: Pairwise-to-Binary (Source + 2 Edited Images)")

if __name__ == "__main__":
    process_all_metrics_consensus()
