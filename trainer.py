import torch
from transformers import Trainer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define Trainer
class TranslationTrainer(Trainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss using the model's forward method.
        """
        # Call the model's forward method
        outputs = model(**inputs)

        # Extract the logits and the labels
        logits = outputs.logits
        labels = inputs.get("labels")

        # Move labels to the same device as logits
        if labels is not None:
            labels = labels.to(logits.device)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.model
        model.eval()

        eval_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs in eval_dataloader:
                # Move inputs to device
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits
                eval_loss += loss.item()
                num_batches += 1

                # Generate predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy())

        eval_loss /= num_batches

        # Compute BLEU score
        smooth = SmoothingFunction().method4
        bleu_scores = []
        for pred, label in zip(all_predictions, all_labels):
            pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
            label_text = self.tokenizer.decode(label, skip_special_tokens=True)
            bleu_score = sentence_bleu([label_text.split()], pred_text.split(), smoothing_function=smooth)
            bleu_scores.append(bleu_score * 100)

        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

        metrics = {
            f'{metric_key_prefix}_loss': eval_loss,
            f'{metric_key_prefix}_bleu_score': avg_bleu_score
        }

        self.log(metrics)

        return metrics