# NLP_Song2Mood

Song lyric emotion classifier for four moods (Angry, Happy, Relaxed, Sad) built with classic NLP baselines and a fine-tuned DistilBERT. Everything runs from `music.ipynb` using the NJU MusicMood dataset.

## Dataset
- Location: `NJU_MusicMood_v1.0/` with class folders (`Angry`, `Happy`, `Relaxed`, `Sad`), each split into `Train/` and `Test/`.
- Size: 400 train (100 per class) and 377 test (Angry 71, Happy 106, Relaxed 101, Sad 99).
- Cleaning (see `clean_lyrics` in the notebook): remove timestamps like `[00:29]`, lowercase, normalize quotes, strip repeated dots/underscores, keep alphanumerics/apostrophes/spaces, collapse whitespace.

## Project Layout
- `music.ipynb` — data loading/cleaning, baselines (BoW, TF-IDF, Word2Vec), segmentation tests, DistilBERT fine-tuning, inference demo.
- `distilbert_model/` — saved DistilBERT checkpoint for inference.
- `NJU_MusicMood_v1.0/` — dataset.

## Setup
### venv (Python 3.10+; tested on 3.11)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### conda
```bash
conda create -n song2mood python=3.11.7 -y
conda activate song2mood
pip install -r requirements.txt
python -m ipykernel install --user --name song2mood --display-name "Python 3.11.7 (song2mood)"
```

## Running `music.ipynb`
1) Activate your environment and open the notebook.
2) Run cells in order:
   - Baselines: Bag-of-Words + LR, TF-IDF + LR (tweak `ngram_range`, e.g., `(1, 3)` for uni/bi/tri), Word2Vec + LR (pretrained gensim vectors averaged per lyric).
   - Segmentation: evaluate start/middle/end lyric slices without retraining.
   - DistilBERT: fine-tune with Hugging Face Trainer; checkpoint saved to `distilbert_model/`.
   - Inference demo: `transformers.pipeline` to classify custom lyrics.

## Results (test set, macro metrics)
Printed by `print_results` in the notebook:

| Model                      | Macro F1 | Accuracy |
| -------------------------- | -------- | -------- |
| Bag-of-Words + LR          | 0.373    | 0.363    |
| TF-IDF + LR                | 0.476    | 0.469    |
| Word2Vec + LR              | 0.455    | 0.454    |
| DistilBERT (full lyric)    | 0.517    | 0.504    |
| DistilBERT (middle slice)  | 0.511    | 0.491    |
| DistilBERT (end slice)     | 0.507    | 0.491    |
| Word2Vec (middle slice)    | 0.431    | 0.435    |

## Quick inference with saved DistilBERT
```python
from transformers import pipeline
clf = pipeline("text-classification", model="distilbert_model", tokenizer="distilbert_model")
clf("I'm dancing on sunshine and nothing can bring me down!")
```

## Dependencies
See `requirements.txt` (numpy, pandas, scikit-learn, datasets, transformers, torch, gensim, accelerate, notebook, ipykernel, ipywidgets). Install via `pip install -r requirements.txt`.
