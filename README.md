# Mood Detector

A small text-classification demo: a scikit-learn TF–IDF/logistic-regression
pipeline and a Streamlit interface. It assigns a mood label and shows ranked
class probabilities. It does not add, restore or rewrite punctuation.

## Implemented behavior

`train_baseline.py` reads `data.csv` (`text,label`), removes missing rows, extracts
word unigrams/bigrams with English stop words removed, and trains balanced
logistic regression. It prints five-fold stratified macro-F1 and a classification
report from a separate 80/20 split, then saves that split's trained pipeline to
`mood_baseline.joblib`.

The committed CSV has 300 examples: 50 each for anger, fear, joy, love, sadness
and surprise. `app.py` loads the saved pipeline, predicts one label, and displays
up to five probabilities for the entered message. It makes no external model API
call and does not persist messages in application code.

## Local setup

The pinned dependencies target **Python 3.12**; the verified environment is Linux
x86_64. `requirements.in` lists direct dependencies and `requirements.txt` pins the
resolved transitive versions (resolved with uv 0.12.15).

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.txt
source .venv/bin/activate
python train_baseline.py
streamlit run app.py --server.address 127.0.0.1
```

Run from the repository root. Training overwrites `mood_baseline.joblib`;
use a scratch checkout if you need to preserve the committed model. Retrain with
the pinned environment before launching: the older serialized model has no
recorded dependency versions and scikit-learn artifacts are version-sensitive.
Only load trusted joblib files; deserialization can execute code. No hosted demo
or deployment is provided by these instructions.

## Limitations

- The dataset is a small collection of short examples; its origin, collection
  method, license and representativeness are not documented. Do not describe it
  as a validated benchmark or redistribute it under an invented license.
- Random splits within this dataset do not establish performance on new users,
  languages, domains, sarcasm or mixed emotions. Five-fold scores and the holdout
  report are diagnostics from the same dataset, not independent external evidence.
- Probabilities are model outputs, not calibrated confidence in a person's
  emotional state. The model cannot diagnose mental health or determine intent.
- The app's fallback for models without `predict_proba` normalizes decision
  scores; those values would not be calibrated probabilities. The trained
  logistic-regression pipeline does provide `predict_proba`.
- The UI assumes at least five classes (the bundled dataset has six), and model
  loading reports a generic missing-file message for any load error. Training
  uses a fixed split seed, but environment pinning is not a quality guarantee.

## Possible next work

Document dataset provenance, add a separate evaluation set and analyze errors
before extending the interface. Punctuation restoration is a separate future
feature requiring its own implementation and evaluation.
