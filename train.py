from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch

from transformers import TrainingArguments
from datasets import load_dataset
import wandb
from transformers import AutoTokenizer
from multi_model import *
from data_processing import *
from trainer import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

custom_model = SMT5Model().to(device)
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

dataset = load_dataset('wanhin/luat-translate', split='train')
src_texts = dataset['query']
tgt_texts = dataset['positive']

# Split dataset into training and validation
train_src_texts, val_src_texts, train_tgt_texts, val_tgt_texts = train_test_split(
    src_texts, tgt_texts, test_size=0.0002, random_state=42)

# Tạo Dataset cho dữ liệu huấn luyện và kiểm tra
train_dataset = Build_Dataset(train_src_texts, train_tgt_texts, tokenizer)
val_dataset = Build_Dataset(val_src_texts, val_tgt_texts, tokenizer)

training_args = TrainingArguments(
    output_dir='./translation-v0-30e',
    run_name='translation-v0',
    evaluation_strategy="epoch",  # Đánh giá mỗi epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Batch size
    per_device_eval_batch_size=4,   # Batch size
    num_train_epochs=30,  # Số lượng epoch
    save_total_limit=2,
    fp16=False,
    gradient_accumulation_steps=32, 
    logging_dir='./logs',
    logging_steps=128,
    report_to="wandb",  # Báo cáo kết quả lên wandb
    load_best_model_at_end=True,  # Load model tốt nhất vào cuối quá trình huấn luyện
)



# Định nghĩa DataCollator
data_collator = DataCollatorForTranslation()

trainer = TranslationTrainer(
    model=custom_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Đăng nhập vào wandb
wandb.login(key='7ac28caf9e3dc3e0685c97df182d52e13a81e311')

# Đăng ký wandb
wandb.init(project="se-form-model")

try:
    # Train the model
    trainer.train()
finally:
    # Kết thúc phiên làm việc với wandb
    wandb.finish()
