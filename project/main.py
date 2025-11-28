import argparse

from baselines import bag_of_words_logreg, most_frequent_class, tfidf_logreg, word2vec_logreg, print_per_class, print_results
from data_loader import EMOTIONS, LABEL2ID, build_hf_dataset, get_lyrics_and_labels
from distilbert_benchmark import train_and_eval_transformer
from position_experiments import get_segment, run_position_tfidf, train_segment_model


def run_baselines():
    train_texts, train_labels = get_lyrics_and_labels("Train")
    dev_texts, dev_labels = get_lyrics_and_labels("Test")

    print("=== Baseline: Bag of Words & Logistic Regression ===")
    preds, metrics, _, _ = bag_of_words_logreg(train_texts, train_labels, dev_texts, dev_labels)
    print_results(metrics)
    print_per_class(dev_labels, preds)

    print("\n=== Baseline: TF-IDF & Logistic Regression ===")
    preds, metrics, _, _ = tfidf_logreg(train_texts, train_labels, dev_texts, dev_labels)
    print_results(metrics)
    print_per_class(dev_labels, preds)

    print("\n=== Word2Vec & Logistic Regression ===")
    preds, metrics, _, _ = word2vec_logreg(train_texts, train_labels, dev_texts, dev_labels)
    print_results(metrics)
    print_per_class(dev_labels, preds)

    print("\n=== Baseline: Most Frequent Class ===")
    train_ids = [LABEL2ID[l] for l in train_labels]
    dev_ids = [LABEL2ID[l] for l in dev_labels]
    preds, metrics, majority_label = most_frequent_class(train_ids, dev_ids)
    print(f"Most frequent class id: {majority_label}")
    print_results(metrics)


def run_position_experiments():
    train_texts, train_labels = get_lyrics_and_labels("Train")
    dev_texts, dev_labels = get_lyrics_and_labels("Test")
    train_ids = [LABEL2ID[l] for l in train_labels]
    dev_ids = [LABEL2ID[l] for l in dev_labels]

    print("=== TF-IDF with START/MID/END position tags ===")
    preds, metrics, _, _ = run_position_tfidf(train_texts, dev_texts, train_ids, dev_ids)
    print_results(metrics)
    print_per_class(dev_ids, preds)

    # Segment-only models
    train_start = [get_segment(t, "start") for t in train_texts]
    train_middle = [get_segment(t, "middle") for t in train_texts]
    train_end = [get_segment(t, "end") for t in train_texts]

    dev_start = [get_segment(t, "start") for t in dev_texts]
    dev_middle = [get_segment(t, "middle") for t in dev_texts]
    dev_end = [get_segment(t, "end") for t in dev_texts]

    for name, tr, dv in [
        ("BEGINNING ONLY", train_start, dev_start),
        ("MIDDLE ONLY", train_middle, dev_middle),
        ("END ONLY", train_end, dev_end),
    ]:
        print(f"\n=== {name} ===")
        preds, metrics, _, _ = train_segment_model(tr, dv, train_ids, dev_ids, name=name)
        print_results(metrics)


def run_transformer(model_name: str, epochs: int, learning_rate: float, batch_size: int):
    train_texts, train_labels = get_lyrics_and_labels("Train")
    dev_texts, dev_labels = get_lyrics_and_labels("Test")
    train_ds, dev_ds = build_hf_dataset(train_texts, train_labels, dev_texts, dev_labels)

    set_pad_token_eos = "gpt2" in model_name
    trainer, eval_results, pred_labels = train_and_eval_transformer(
        model_name=model_name,
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        output_dir=f"./{model_name.replace('/', '_')}_output",
        num_epochs=epochs,
        learning_rate=learning_rate,
        train_bs=batch_size,
        eval_bs=batch_size,
        set_pad_token_eos=set_pad_token_eos,
    )
    print(f"{model_name} eval: {eval_results}")
    print("Example predictions:", pred_labels[:5])
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Song2Mood experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("baselines", help="Run classical baselines")
    subparsers.add_parser("position", help="Run position tagging experiments")

    distil_parser = subparsers.add_parser("distilbert", help="Fine-tune a transformer model")
    distil_parser.add_argument("--model", default="distilbert/distilbert-base-uncased", help="Model name on Hugging Face")
    distil_parser.add_argument("--epochs", type=int, default=3)
    distil_parser.add_argument("--lr", type=float, default=5e-5)
    distil_parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "baselines":
        run_baselines()
    elif args.command == "position":
        run_position_experiments()
    elif args.command == "distilbert":
        run_transformer(args.model, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
