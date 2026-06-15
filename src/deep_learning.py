"""Deep learning extension (Part 11): LSTM / GRU / Transformer.

This module is fully optional. PyTorch is imported lazily and guarded — if it
is not installed, :data:`TORCH_AVAILABLE` is False and :func:`train_sequence_model`
raises a clear, actionable error instead of crashing on import. The rest of the
project (preprocessing, CBCB labelling, the sklearn models, the Streamlit app)
works without torch.

Task formulation
----------------
We frame next-genre prediction as sequence classification: for each user we
build a sliding window of their last ``seq_len`` genre codes and predict the
genre of the following interaction. Three architectures share one train/eval
loop:

    LSTMRecommender         embedding -> LSTM -> linear
    GRURecommender          embedding -> GRU  -> linear
    TransformerRecommender  embedding + positional -> TransformerEncoder -> linear
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config
from .utils import LOG

# --------------------------------------------------------------------------- #
# Guarded torch import
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False

_INSTALL_HINT = (
    "PyTorch is not installed. The deep-learning extension is optional; "
    "install it with `pip install torch` to enable LSTM/GRU/Transformer models."
)


# --------------------------------------------------------------------------- #
# Sequence construction
# --------------------------------------------------------------------------- #
def build_sequences(
    df: pd.DataFrame, seq_len: int = 5
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build (X, y, vocab_size) sliding windows of genre codes per user.

    X[i] is a length-``seq_len`` array of genre codes; y[i] is the next code.
    """
    df = df.sort_values(["User_ID", "Date"])
    codes = df["Program_Genre"].astype("category")
    vocab = list(codes.cat.categories)
    code_by_user = (
        df.assign(_code=codes.cat.codes).groupby("User_ID")["_code"].apply(list)
    )

    X: list[list[int]] = []
    y: list[int] = []
    for seq in code_by_user:
        if len(seq) <= seq_len:
            continue
        for i in range(len(seq) - seq_len):
            X.append(seq[i : i + seq_len])
            y.append(seq[i + seq_len])

    return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64), len(vocab)


# --------------------------------------------------------------------------- #
# Model definitions (only defined if torch is available)
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class LSTMRecommender(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.LSTM(embed_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            out, _ = self.rnn(self.embed(x))
            return self.head(out[:, -1, :])

    class GRURecommender(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.GRU(embed_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            out, _ = self.rnn(self.embed(x))
            return self.head(out[:, -1, :])

    class TransformerRecommender(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 32,
                     nhead: int = 4, layers: int = 2, seq_len: int = 5):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, batch_first=True,
                dim_feedforward=embed_dim * 4,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            h = self.embed(x) + self.pos[:, : x.size(1), :]
            h = self.encoder(h)
            return self.head(h[:, -1, :])

    _ARCHITECTURES = {
        "lstm": LSTMRecommender,
        "gru": GRURecommender,
        "transformer": TransformerRecommender,
    }


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_sequence_model(
    df: pd.DataFrame,
    arch: str = "lstm",
    seq_len: int = 5,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train one sequence model and return a small metrics dict.

    Raises
    ------
    RuntimeError if PyTorch is not installed (with an install hint).
    ValueError   if ``arch`` is unknown.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(_INSTALL_HINT)
    if arch not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(_ARCHITECTURES)}.")

    torch.manual_seed(config.RANDOM_SEED)
    X, y, vocab_size = build_sequences(df, seq_len=seq_len)
    if len(X) == 0:
        raise ValueError("Not enough per-user history to build sequences.")

    n_train = int(len(X) * (1 - config.TEST_SIZE))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    kwargs = {"vocab_size": vocab_size}
    if arch == "transformer":
        kwargs["seq_len"] = seq_len
    model = _ARCHITECTURES[arch](**kwargs)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            total += loss.item() * len(xb)
        avg = total / len(train_ds)
        history.append(avg)
        LOG.info("[%s] epoch %d/%d loss=%.4f", arch, epoch + 1, epochs, avg)

    # Test accuracy
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(Xte)).argmax(dim=1).numpy()
    acc = float((preds == yte).mean()) if len(yte) else float("nan")
    LOG.info("[%s] test top-1 accuracy=%.3f", arch, acc)

    return {
        "architecture": arch,
        "vocab_size": vocab_size,
        "n_sequences": int(len(X)),
        "test_accuracy": acc,
        "loss_history": history,
        "model": model,
    }


if __name__ == "__main__":
    from .dataset_generator import generate_dataset

    if not TORCH_AVAILABLE:
        print(_INSTALL_HINT)
    else:
        demo = generate_dataset(n_rows=8_000, n_users=300)
        for a in ("lstm", "gru", "transformer"):
            res = train_sequence_model(demo, arch=a, epochs=2)
            print(f"{a}: acc={res['test_accuracy']:.3f}")
