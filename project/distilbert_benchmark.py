import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from data_loader import EMOTIONS, ID2LABEL, LABEL2ID


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc}


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 256):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(_tok, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch")
    return tokenized


def train_and_eval_transformer(
    model_name: str,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    train_bs: int = 4,
    eval_bs: int = 4,
    set_pad_token_eos: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if set_pad_token_eos:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenize_dataset(dev_dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTIONS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    if set_pad_token_eos:
        model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_steps=16,
        log_level="error",
        report_to="none",
        save_strategy="epoch",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    pred_output = trainer.predict(tokenized_dev)
    logits = pred_output.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_labels = [ID2LABEL[i] for i in pred_ids]

    return trainer, eval_results, pred_labels


__all__ = ["train_and_eval_transformer", "tokenize_dataset", "compute_metrics"]
