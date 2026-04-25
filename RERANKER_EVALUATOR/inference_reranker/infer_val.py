import torch
import json
import os
import re # Thêm thư viện re để trích xuất text
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

# --- 1. CẤU HÌNH ---
METRIC_NAME = "All_Metrics_Fidelity"
MODEL_PATH = "/datastore/raccoon/truongtq/MODELS/Qwen3-VL-Reranker-8B"
LORA_PATH  = "/datastore/raccoon/truongtq/TRAINING_LORA_VLRERANKER/outputs/lora_fidelity_lorarank128_lr2e-5"
VAL_JSONL  = "/datastore/raccoon/truongtq/USER_STUDY/ScoringAndRanking/user_study_reranker_test.jsonl"

now = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"/datastore/raccoon/truongtq/TRAINING_LORA_VLRERANKER/inference_reranker/results_infer/val_set/{METRIC_NAME}_lorarank64_8B_{now}.json"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. LOAD MODEL & ADAPTER ---
print(f"--- Loading Model & Adapter ---")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

YES_WORDS = ["yes", "Yes", " yes", " Yes"]
NO_WORDS = ["no", "No", " no", " No"]
YES_IDS = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in YES_WORDS]
NO_IDS = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in NO_WORDS]

# --- 3. VÒNG LẶP ĐÁNH GIÁ ---
y_true, y_pred, results_log = [], [], []

with open(VAL_JSONL, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"--- Starting Pairwise Inference on {len(lines)} samples ---")

for line in tqdm(lines):
    data = json.loads(line)
    
    gt_text = data['conversations'][1]['value']
    gt_label = 1 if gt_text.lower() == "yes" else 0
    
    system_prompt = data['system']
    user_prompt = data['conversations'][0]['value']
    img_paths = data['images'] # [source_img, candidate1_img, candidate2_img]

    # --- TRÍCH XUẤT INSTRUCTION ---
    # Tìm đoạn text nằm giữa "Edit Instruction: " và " Task: "
    instr_match = re.search(r"Edit Instruction: (.*?) Task:", user_prompt)
    instruction = instr_match.group(1).strip() if instr_match else "Unknown"
    
    # --- CẮT GHÉP PROMPT ---
    parts = user_prompt.split('<image>')
    user_content = []
    user_content.append({"type": "text", "text": parts[0]})
    user_content.append({"type": "image", "image": f"file://{img_paths[0]}"})
    user_content.append({"type": "text", "text": parts[1]})
    user_content.append({"type": "image", "image": f"file://{img_paths[1]}"})
    user_content.append({"type": "text", "text": parts[2]})
    user_content.append({"type": "image", "image": f"file://{img_paths[2]}"})
    
    if len(parts) > 3 and parts[3].strip() != "":
        user_content.append({"type": "text", "text": parts[3]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content}
    ]

    # Preprocessing
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(device)

    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)

    with torch.no_grad():
        # --- THIẾT LẬP ĐỂ TẮT NGẪU NHIÊN (GREEDY SEARCH) ---
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2, 
            do_sample=False,           # QUAN TRỌNG: Tắt lấy mẫu ngẫu nhiên
            # temperature=0,           # Không bắt buộc nếu do_sample=False nhưng có thể thêm
            # top_p=1.0,               # Lấy toàn bộ phân phối
            num_beams=1,               # Đảm bảo không dùng beam search (mặc định là 1)
            return_dict_in_generate=True, 
            output_scores=True
        )
        
        # 1. TRÍCH XUẤT TEXT
        gen_ids = outputs.sequences
        gen_text = processor.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        
        # 2. TÍNH LOGIT SCORE
        # Khi do_sample=False, logits trả về vẫn là giá trị thô ổn định
        first_step_logits = outputs.scores[0][0]
        
        yes_logit = max([first_step_logits[tid].item() for tid in YES_IDS])
        no_logit  = max([first_step_logits[tid].item() for tid in NO_IDS])
        
        score = torch.sigmoid(torch.tensor(yes_logit - no_logit)).item()
        pred_label = 1 if "yes" in gen_text else 0

    y_true.append(gt_label)
    y_pred.append(pred_label)
    
    results_log.append({
        "id": data['id'],
        "instruction": instruction, # BỔ SUNG TRƯỜNG NÀY
        "source_path": img_paths[0],
        "candidate1_path": img_paths[1],
        "candidate2_path": img_paths[2],
        "ground_truth": "yes" if gt_label == 1 else "no",
        "prediction": "yes" if pred_label == 1 else "no",
        "score": score,
        "match": gt_label == pred_label
    })

# --- 4. TÍNH TOÁN VÀ LƯU ---
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)

print("\n" + "="*45)
print(f"PAIRWISE EVALUATION RESULTS: {METRIC_NAME}")
print(f"Total samples: {len(y_true)}")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("="*45)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": {
            "metric_name": METRIC_NAME,
            "lora_path": LORA_PATH,
            "total_samples": len(y_true),
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        },
        "details": results_log
    }, f, indent=4, ensure_ascii=False)

print(f"✅ Full report saved to {OUTPUT_FILE}")