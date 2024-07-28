from torch.utils.data import Dataset

# Định nghĩa một lớp Dataset cho dữ liệu
class Build_Dataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_encoding = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        tgt_encoding = self.tokenizer(tgt_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        labels = tgt_encoding.input_ids.clone()


        return {
            'input_ids': src_encoding.input_ids.squeeze(),
            'attention_mask': src_encoding.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

# Định nghĩa lớp DataCollator
class DataCollatorForTranslation:
    def __call__(self, features):
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }