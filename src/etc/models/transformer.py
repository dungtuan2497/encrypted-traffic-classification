"""Small ET-BERT-style byte Transformer (review Section 5.3.1), scaled down for CPU/laptop.

Tokenisation: one token per byte (vocab 256) + [PAD], [CLS], [SEP], [MASK] = 260.
A session becomes  [CLS] bytes(pkt1) [SEP] bytes(pkt2) [SEP] ...  (max 512 tokens).
NOTE: ET-BERT uses byte *bigrams* (vocab 65,536) and a much larger BERT-base encoder;
single bytes keep the embedding table tiny so the model trains on a laptop.

Embeddings = token + learned position + packet-index (which packet a byte belongs to,
analogous to ET-BERT's burst/segment structure). Two heads:
  * MLM head (pre-training, Masked Byte Modeling)
  * classification head on the [CLS] hidden state (fine-tuning)
The model consumes a single int tensor x of shape (B, 2, L): x[:, 0] = token ids,
x[:, 1] = packet indices, so it plugs into the generic training loop unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_BYTES = 256
PAD, CLS, SEP, MASK = 256, 257, 258, 259
VOCAB_SIZE = 260


class ByteTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 512,
        max_packets: int = 32,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.pkt_emb = nn.Embedding(max_packets + 1, d_model)  # 0 = [CLS]/padding
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size)
        )
        self.cls_head = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes)) if num_classes else None
        )

    @staticmethod
    def trim(x: torch.Tensor, multiple: int = 64) -> torch.Tensor:
        """Drop trailing all-padding columns of a batch (saves compute on short sessions).

        The kept length is rounded up to a multiple of 64 so the accelerator only ever sees
        a handful of distinct shapes (MPS recompiles kernels for every new shape).
        """
        length = int((x[:, 0] != PAD).sum(dim=1).max())
        length = min(x.shape[-1], max(multiple, -(-length // multiple) * multiple))
        return x[..., :length]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, pkt_idx = x[:, 0].long(), x[:, 1].long()
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        h = self.tok_emb(tokens) + self.pos_emb(pos)[None] + self.pkt_emb(pkt_idx)
        h = self.emb_drop(self.emb_norm(h))
        pad_mask = tokens == PAD
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.final_norm(h), pad_mask

    def mlm_logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, L) -> (B, L, vocab) logits for masked byte prediction."""
        h, _ = self.encode(x)
        return self.mlm_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification logits from the [CLS] position."""
        if self.cls_head is None:
            raise RuntimeError("Model built without a classification head")
        h, _ = self.encode(self.trim(x))
        return self.cls_head(h[:, 0])

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights shared between pre-training and fine-tuning (everything but cls_head)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("cls_head.")}


def build_transformer(mcfg: dict, num_classes: int | None) -> ByteTransformer:
    return ByteTransformer(
        num_classes=num_classes,
        vocab_size=int(mcfg["vocab_size"]),
        max_len=int(mcfg["max_len"]),
        max_packets=int(mcfg["max_packets"]),
        d_model=int(mcfg["d_model"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        ffn_dim=int(mcfg["ffn_dim"]),
        dropout=float(mcfg["dropout"]),
    )


def mask_tokens(
    tokens: torch.Tensor,
    mask_prob: float = 0.15,
    mask_token_ratio: float = 0.8,
    random_token_ratio: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking on byte tokens only (special tokens are never selected).

    Of the selected 15%: 80% -> [MASK], 10% -> random byte, 10% unchanged.
    Returns (masked tokens, labels) with labels = -100 for unselected positions.
    """
    dev = tokens.device
    rand = lambda: torch.rand(tokens.shape, generator=generator, device="cpu").to(dev)  # noqa: E731
    candidates = tokens < NUM_BYTES
    selected = (rand() < mask_prob) & candidates
    labels = torch.where(selected, tokens, torch.full_like(tokens, -100))
    r = rand()
    to_mask = selected & (r < mask_token_ratio)
    to_random = selected & (r >= mask_token_ratio) & (r < mask_token_ratio + random_token_ratio)
    out = tokens.clone()
    out[to_mask] = MASK
    random_bytes = torch.randint(0, NUM_BYTES, tokens.shape, generator=generator).to(dev)
    out[to_random] = random_bytes[to_random]
    return out, labels
