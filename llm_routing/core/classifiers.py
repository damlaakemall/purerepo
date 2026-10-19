"""Query classifiers that learn ``p(x) in [0, 1]`` from *soft* labels.

Many routers boil down to "a text classifier trained on cleverly constructed
labels". The labels are the paper-specific part; the classifier is reusable.
Hybrid LLM uses soft labels (probabilities, not 0/1), trained with binary
cross-entropy:

    BCE(y, p) = -[ y log p + (1 - y) log(1 - p) ]       with y in [0, 1]

Two backends:

* ``TfidfLogisticClassifier`` — CPU, seconds to train, no GPU needed. Good for
  learning/experimenting and as a baseline.
* ``TransformerClassifier`` — the paper's setup (DeBERTa-v3-large, ~300M
  params). Needs ``torch`` + ``transformers``; a GPU is strongly recommended.

``fit`` optionally takes a validation set. The transformer then keeps the epoch with
the lowest validation BCE (Hybrid LLM §4.1: "use the validation set to choose the
best checkpoints"); TF-IDF has no epochs and ignores it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SoftLabelClassifier(ABC):
    @abstractmethod
    def fit(self, texts: list[str], soft_labels: np.ndarray, val_texts: list[str] | None = None,
            val_labels: np.ndarray | None = None) -> "SoftLabelClassifier": ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


class TfidfLogisticClassifier(SoftLabelClassifier):
    """TF-IDF features + logistic regression, trained on soft labels.

    scikit-learn's LogisticRegression only accepts hard 0/1 targets, so we use
    an exact trick: duplicate every example, once with label 1 and weight y,
    once with label 0 and weight (1 - y). The weighted log-loss of the pair is
    exactly the soft-label BCE above.
    """

    def __init__(self, C: float = 1.0, max_features: int = 50_000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, texts, soft_labels, val_texts=None, val_labels=None):
        y = np.clip(np.asarray(soft_labels, dtype=float), 0.0, 1.0)
        X = self.vectorizer.fit_transform(texts)
        from scipy.sparse import vstack

        X2 = vstack([X, X])
        y2 = np.concatenate([np.ones(len(y)), np.zeros(len(y))])
        w2 = np.concatenate([y, 1.0 - y])
        keep = w2 > 0
        if len(np.unique(y2[keep])) < 2:
            # all labels identical: the classifier can only learn a constant
            self._constant = float(y.mean())
            return self
        self._constant = None
        self.model.fit(X2[keep], y2[keep], sample_weight=w2[keep])
        return self

    def predict_proba(self, texts):
        if getattr(self, "_constant", None) is not None:
            return np.full(len(texts), self._constant)
        return self.model.predict_proba(self.vectorizer.transform(texts))[:, 1]


class TransformerClassifier(SoftLabelClassifier):
    """Fine-tune an encoder (default: DeBERTa-v3-large, as in the paper) with soft-label BCE.

    Architecture: encoder -> [CLS] embedding -> linear head -> 1 logit -> sigmoid.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        lr: float = 1e-5,
        epochs: int = 5,
        batch_size: int = 16,
        max_length: int = 512,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        device: str | None = None,
        seed: int = 0,
    ):
        self.model_name = model_name
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.max_length, self.weight_decay, self.warmup_ratio = max_length, weight_decay, warmup_ratio
        self.device, self.seed = device, seed

    def _setup(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1).to(self.device)

    def fit(self, texts, soft_labels, val_texts=None, val_labels=None):
        import torch
        from transformers import get_linear_schedule_with_warmup

        self._setup()
        y = torch.tensor(np.clip(soft_labels, 0, 1), dtype=torch.float32)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_ratio * steps), steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()  # accepts soft targets directly
        g = np.random.default_rng(self.seed)
        best, self.best_val_bce_, self.best_epoch_ = None, float("inf"), self.epochs
        self.model.train()
        for epoch in range(self.epochs):
            perm = g.permutation(len(texts))
            total = 0.0
            for b in range(0, len(texts), self.batch_size):
                idx = perm[b : b + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits, y[idx].to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                total += loss.item() * len(idx)
            msg = f"[TransformerClassifier] epoch {epoch + 1}/{self.epochs} loss={total / len(texts):.4f}"
            if val_texts is not None and len(val_texts):
                val_loss = self.bce(val_texts, val_labels)
                msg += f" val_bce={val_loss:.4f}"
                if val_loss < self.best_val_bce_:  # checkpoint selection on val only
                    self.best_val_bce_, self.best_epoch_ = val_loss, epoch + 1
                    best = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.train()
            print(msg)
        if best is not None:
            self.model.load_state_dict(best)
        return self

    def bce(self, texts, soft_labels) -> float:
        """Soft-label BCE of the current model on ``texts`` (used for checkpoint selection)."""
        p = np.clip(self.predict_proba(texts), 1e-7, 1 - 1e-7)
        y = np.clip(np.asarray(soft_labels, dtype=float), 0, 1)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def predict_proba(self, texts):
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(texts), self.batch_size * 4):
                enc = self.tokenizer(texts[b : b + self.batch_size * 4], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                out.append(torch.sigmoid(self.model(**enc).logits.squeeze(-1)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def make_classifier(backend: str, **kwargs) -> SoftLabelClassifier:
    if backend == "tfidf":
        return TfidfLogisticClassifier(**kwargs)
    if backend in ("deberta", "transformer"):
        return TransformerClassifier(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")
