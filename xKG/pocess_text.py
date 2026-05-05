import json

str1 = """
import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class ContrastPairConstructor:
    \"\"\"
    Constructs tokenized contrast pairs (positive, negative) for binary (yes/no) questions.

    Attributes:
        tokenizer: Huggingface tokenizer to encode text.
        model_type: One of {"encoder", "decoder", "encoder_decoder"}.
        use_decoder: If True and model_type=="encoder_decoder", feeds question to encoder
                     and answer to decoder; else feeds full prompt to encoder.
        max_length: Maximum token length for truncation/padding.
    \"\"\"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_type: str = "encoder_decoder",
        use_decoder: bool = False,
        max_length: int = 512,
    ) -> None:
        \"\"\"
        Args:
            tokenizer: Pretrained Huggingface tokenizer.
            model_type: "encoder", "decoder", or "encoder_decoder".
            use_decoder: Only for encoder-decoder models.
            max_length: Sequence length limit.
        \"\"\"
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_decoder = use_decoder
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            # assign pad token if missing
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore

    def _encode_encoder(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def _encode_decoder(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text + self.tokenizer.eos_token,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def _encode_encoder_decoder(
        self, question: str, answer: str
    ) -> Dict[str, torch.Tensor]:
        if self.use_decoder:
            enc = self.tokenizer(
                question,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            dec = self.tokenizer(
                answer,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                question,
                answer,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            dec = self.tokenizer(
                "",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "decoder_input_ids": dec["input_ids"].squeeze(0),
            "decoder_attention_mask": dec["attention_mask"].squeeze(0),
        }

    def construct(
        self, question: str, positive_label: str = "Yes", negative_label: str = "No"
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        \"\"\"
        Build tokenized inputs for positive ("Yes") and negative ("No") statements.

        Returns:
            pos_batch, neg_batch: dicts with tensors for model inputs.
        \"\"\"
        pos_text = f"{question} {positive_label}"
        neg_text = f"{question} {negative_label}"

        if self.model_type == "encoder":
            pos_batch = self._encode_encoder(pos_text)
            neg_batch = self._encode_encoder(neg_text)
        elif self.model_type == "decoder":
            pos_batch = self._encode_decoder(pos_text)
            neg_batch = self._encode_decoder(neg_text)
        else:  # encoder_decoder
            pos_batch = self._encode_encoder_decoder(question, positive_label)
            neg_batch = self._encode_encoder_decoder(question, negative_label)

        return pos_batch, neg_batch


def load_model_and_tokenizer(
    model_name: str, device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    \"\"\"
    Loads a pretrained Huggingface model and tokenizer.

    Args:
        model_name: e.g. "bert-base-uncased".
        device: "cuda"/"cpu". If None, auto-select.
    Returns:
        model (eval mode), tokenizer.
    \"\"\"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_last_token_indices(attn_mask: torch.LongTensor) -> torch.LongTensor:
    \"\"\"
    For each sequence, return index of last non-padding token.
    \"\"\"
    lengths = attn_mask.sum(dim=1)
    return torch.clamp(lengths - 1, min=0).long()


def extract_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: List[Dict[str, torch.Tensor]],
    layer: int = -1,
) -> np.ndarray:
    \"\"\"
    Given a list of tokenized inputs (dicts), run through model and
    extract hidden state of specified layer at the last real token.

    Args:
        inputs: list of dicts with "input_ids" and "attention_mask", optionally
                "decoder_input_ids" and "decoder_attention_mask".
        layer: which hidden layer to use.
    Returns:
        feats: array of shape (n, hidden_size).
    \"\"\"
    device = next(model.parameters()).device
    all_feats: List[np.ndarray] = []
    for batch in inputs:
        # organize batch of size 1
        batch_t = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch_t, output_hidden_states=True)
        # choose hidden_states key
        hs = out.hidden_states[layer]  # (1, seq_len, d)
        attn = batch_t.get("attention_mask", batch_t.get("decoder_attention_mask"))
        pos = get_last_token_indices(attn)
        idx = pos.item()
        feat = hs[0, idx, :].cpu().numpy()  # (d,)
        all_feats.append(feat)
    return np.stack(all_feats, axis=0)


def normalize_features(
    X_plus: np.ndarray, X_minus: np.ndarray, scale: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"
    Mean-normalize positive and negative features.

    Args:
        X_plus: (n, d), features ending in "Yes"
        X_minus: (n, d), features ending in "No"
        scale: if True, divide by mean norm * sqrt(d).
    Returns:
        Vp, Vn: normalized arrays (n, d).
    \"\"\"
    mu_p = X_plus.mean(axis=0, keepdims=True)
    mu_n = X_minus.mean(axis=0, keepdims=True)
    Vp = X_plus - mu_p
    Vn = X_minus - mu_n
    if scale:
        d = X_plus.shape[1]
        avg_p = np.linalg.norm(X_plus, axis=1).mean() * math.sqrt(d)
        avg_n = np.linalg.norm(X_minus, axis=1).mean() * math.sqrt(d)
        if avg_p > 0:
            Vp /= avg_p
        if avg_n > 0:
            Vn /= avg_n
    return Vp, Vn


class LinearProbe(nn.Module):
    \"\"\"
    Linear probe: p(x) = sigmoid(θ^T x + b)
    \"\"\"

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # (n,1)
        return self.sigmoid(logits).squeeze(-1)  # (n,)


class CCS:
    \"\"\"
    Contrast-Consistent Search (CCS) for unsupervised truth extraction.

    Trains a linear probe p on contrast pairs (x⁺, x⁻) by minimizing
      L = mean[(p(x⁺) - (1-p(x⁻)))^2 + min(p(x⁺), p(x⁻))^2].
    \"\"\"

    def __init__(
        self,
        X_plus: np.ndarray,
        X_minus: np.ndarray,
        nepochs: int = 1000,
        ntries: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = -1,
        device: Optional[str] = None,
    ) -> None:
        \"\"\"
        Args:
            X_plus: (n, d) features for "Yes" prompts.
            X_minus: (n, d) features for "No" prompts.
            nepochs: epochs per try.
            ntries: random restarts.
            lr: learning rate.
            weight_decay: L2 reg.
            batch_size: if -1, full-batch.
            device: torch device.
        \"\"\"
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # center data
        self.Xp, self.Xn = normalize_features(X_plus, X_minus, scale=False)
        self.n, self.d = self.Xp.shape
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.wd = weight_decay
        self.batch_size = self.n if batch_size < 0 else batch_size

        self.best_probe: nn.Module = None  # to hold best model

    def _to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        xp = torch.tensor(self.Xp, dtype=torch.float32, device=self.device)
        xn = torch.tensor(self.Xn, dtype=torch.float32, device=self.device)
        return xp, xn

    def _ccs_loss(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Consistency + confidence loss.
        \"\"\"
        c1 = ((p - (1 - q)) ** 2).mean()
        c2 = torch.min(p, q).pow(2).mean()
        return c1 + c2

    def _train_once(self) -> float:
        xp, xn = self._to_tensors()
        probe = LinearProbe(self.d).to(self.device)
        opt = torch.optim.AdamW(probe.parameters(), lr=self.lr, weight_decay=self.wd)

        for _ in range(self.nepochs):
            perm = torch.randperm(self.n, device=self.device)
            xp_sh, xn_sh = xp[perm], xn[perm]
            for i in range(0, self.n, self.batch_size):
                bx = xp_sh[i : i + self.batch_size]
                by = xn_sh[i : i + self.batch_size]
                p = probe(bx)
                q = probe(by)
                loss = self._ccs_loss(p, q)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # return final loss and model
        return loss.item(), probe

    def fit(self) -> None:
        \"\"\"
        Runs ntries of training and keeps best probe.
        \"\"\"
        best_loss = float("inf")
        for _ in range(self.ntries):
            loss, model = self._train_once()
            if loss < best_loss:
                best_loss, self.best_probe = loss, copy.deepcopy(model)

    def predict(self, Xp_test: np.ndarray, Xn_test: np.ndarray) -> np.ndarray:
        \"\"\"
        Given test contrast features, returns binary predictions (0/1).
        \"\"\"
        Vp, Vn = normalize_features(Xp_test, Xn_test, scale=False)
        xp = torch.tensor(Vp, dtype=torch.float32, device=self.device)
        xn = torch.tensor(Vn, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p = self.best_probe(xp)
            q = self.best_probe(xn)
        score = 0.5 * (p + (1 - q))
        return (score > 0.5).cpu().numpy().astype(int)
"""

str2 = """
if __name__ == "__main__":
    # Synthetic demonstration of CCS on random features
    np.random.seed(0)
    # create 100 samples of d=16
    Xp = np.random.randn(100, 16) + 1.0  # shifted mean
    Xn = np.random.randn(100, 16) - 1.0  # opposite shift

    ccs = CCS(Xp, Xn, nepochs=200, ntries=3, lr=1e-2, weight_decay=1e-3)
    ccs.fit()

    # split half for test
    Xp_train, Xp_test = Xp[:50], Xp[50:]
    Xn_train, Xn_test = Xn[:50], Xn[50:]
    ccs_tr = CCS(Xp_train, Xn_train, nepochs=200, ntries=3, lr=1e-2, weight_decay=1e-3)
    ccs_tr.fit()
    preds = ccs_tr.predict(Xp_test, Xn_test)
    # ground-truth: positive samples should be class 1
    y_true = np.ones(50, dtype=int)
    acc = (preds == y_true).mean()
    print(f"Synthetic CCS accuracy (yes-class): {{acc:.2f}}")
"""
str3 = """
This module implements Contrast-Consistent Search (CCS), an unsupervised method to extract a latent yes/no classifier from a pretrained transformer’s hidden states. It provides:

1. ContrastPairConstructor: formats and tokenizes binary questions into “Yes”/“No” prompt pairs.
2. load_model_and_tokenizer & extract_hidden_states: load any Huggingface model, run prompts, and extract a chosen layer’s hidden vector at the last token.
3. normalize_features: removes group means (and optional scaling) to eliminate superficial artifacts.
4. LinearProbe: a simple linear + sigmoid mapping from centered features to probabilities.
5. CCS: trains the probe by minimizing a contrast-consistency loss across paired features, with multiple restarts to avoid local minima. It offers `.fit()` and `.predict()` for inference.

All training is self-contained and requires only the raw hidden features.
"""


print(json.dumps(str1))
print("---------------------------------------------------------------------")
print(json.dumps(str2))
print("---------------------------------------------------------------------")
print(json.dumps(str3))
