# Encrypted Traffic Classification — from Machine Learning to Pre-training

An experimental companion to the survey:

> Dong et al., *Deep learning and pre-training technology for encrypted traffic classification: A comprehensive review*, Neurocomputing 617 (2025) 128444.

The survey groups encrypted-traffic classification methods into four generations: plaintext rules → traditional machine learning → deep learning → **pre-training**. This project implements a representative of each of the last three generations on **the same dataset and the same train/val/test split**, and compares them fairly.

> Progress report (in Vietnamese): [`PROGRESS.md`](PROGRESS.md). Original specification (in Vietnamese): [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

---

## 1. Motivation and goals

Most Internet traffic is now encrypted (HTTPS/TLS, VPN), so packet payloads can no longer be inspected. Applications still leave a fingerprint: packet sizes, timing, direction, and the few header bytes that stay in clear text. The goal is to classify a network **session** into the application (Skype, Gmail, FTP, …) or **malware family** (Zeus, Cridex, Neris, …) that produced it, **without decryption**.

| Generation | Model | Input |
|---|---|---|
| Traditional ML | Random Forest (XGBoost optional) | 45 statistical flow features |
| Deep learning | 1D-CNN (Wang et al. 2017) | first 784 bytes of the session |
| Deep learning (optional) | BiGRU (simplified FS-Net) | directed packet-size sequence + inter-arrival times |
| **Pre-training** (focus) | Small ET-BERT-style Transformer (~3.5M parameters) | byte tokens, pre-trained with Masked Byte Modeling |

### Research questions

1. **RQ1 — Generation comparison.** How do traditional ML (Random Forest on statistical features), deep learning (1D-CNN, GRU) and a pre-trained Transformer differ in accuracy, parameter count, training time and inference latency?
2. **RQ2 — Label efficiency.** The survey claims pre-training reduces the dependence on labelled data. We fine-tune with 1%, 5%, 10% and 100% of the labels and compare against the same architecture trained from scratch.
3. **RQ3 — Shortcut learning.** The survey notes that using IP addresses as input invites overfitting. We rerun the byte models with and without masking IP addresses / ports and measure the change in accuracy.

RQ3 is the critical part: it helps assess the ~99% figures that often appear in the survey's comparison tables.

## 2. Scope and assumptions

- Runs on a laptop CPU; a GPU (CUDA or Apple MPS) is used automatically when available. All models are small (the Transformer stays below ~5M parameters).
- Every expensive step has a `--limit` or `--max-sessions-per-class` option for a quick run on a small subset first.
- **No fabricated results.** Every number in a report must come from code that actually ran (files in `results/`). Anything not yet run is reported as "no result yet".
- Order of work: small subset → verify the pipeline → full dataset.

## 3. Data

### Main dataset: USTC-TFC2016

Chosen because it is used in the survey's comparison (Table 14), is moderately sized, and has 20 classes (10 benign, 10 malware).

- Benign: BitTorrent, Facetime, FTP, Gmail, MySQL, Outlook, Skype, SMB, Weibo, WorldOfWarcraft
- Malware: Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus
- Source: <https://github.com/yungshenglu/USTC-TFC2016>. `make download` clones it and extracts the `.7z` archives into `data/raw/ustc-tfc2016/<Group>/<Class>/*.pcap` (≈3.8 GB).
- Caveat for the report: the survey points out that this dataset contains a significant share of plaintext and long flows, so it is a relatively easy task. High scores here do not imply a strong model on real encrypted traffic.

### Optional dataset: ISCX VPN-nonVPN 2016

Can be added later (~28 GB, from the Canadian Institute for Cybersecurity, UNB). The code is dataset-agnostic: a dataset is defined by a single YAML file mapping `pcap glob → label` (see `configs/datasets/ustc_tfc2016.yaml`).

### Mandatory rules against data leakage

1. **Split by session, not by packet.** All packets of a session go to the same split.
2. **De-duplicate before splitting**: hash the preprocessed byte array and drop identical sessions; the number removed is logged.
3. **Pre-training uses the training split only** (unlabelled), never val/test — exactly the mistake the survey identifies in YaTC.
4. **Split once**, save the indices to `data/processed/splits.json`; every model uses this file.
5. Ratio 80/10/10, stratified by label, `seed=42`.

## 4. Tech stack

- Python 3.11
- pcap parsing: `dpkt` (fast, pcap and pcapng); not `scapy` for bulk processing (too slow)
- Data: `numpy`, `pandas`, `pyarrow` (`.parquet`)
- ML: `scikit-learn`, `xgboost` (optional)
- DL: `torch`
- Config: `pyyaml`; CLI: `argparse`
- Plots: `matplotlib`; utilities: `tqdm`, `rich`; tests: `pytest`
- Environment: `venv` + `requirements.txt`

## 5. Project layout

```
.
├── README.md                  # this file
├── PROJECT_GUIDE.md           # original specification (Vietnamese)
├── PROGRESS.md                # progress report (Vietnamese)
├── pyproject.toml / requirements.txt
├── Makefile
├── configs/
│   ├── default.yaml           # paths, seed, preprocessing parameters
│   ├── datasets/              # pcap glob -> label mappings
│   ├── rf.yaml  cnn.yaml  gru.yaml
│   └── transformer.yaml       # pre-training and fine-tuning
├── data/
│   ├── raw/                   # original pcaps (git-ignored)
│   └── processed/             # parquet/npy + splits.json (git-ignored)
├── src/etc/                   # "etc" = encrypted traffic classification
│   ├── config.py  cli.py  utils.py
│   ├── data/
│   │   ├── pcap_reader.py     # streaming pcap reader, session extraction
│   │   ├── features.py        # statistical features and packet-size sequences
│   │   ├── bytes_repr.py      # byte representation, IP/port masking
│   │   ├── build_dataset.py   # assemble dataset, de-dup, split
│   │   ├── processed.py       # shared loaders, label subsampling
│   │   └── torch_datasets.py  # model-specific tensors, tokenizer, samplers
│   ├── models/                # rf.py, cnn1d.py, gru.py, transformer.py
│   ├── train/
│   │   ├── train_sklearn.py
│   │   ├── train_torch.py     # shared training loop for all torch models
│   │   ├── pretrain_mlm.py    # Masked Byte Modeling
│   │   └── finetune.py
│   └── eval/                  # metrics.py, latency.py (report.py: planned)
├── scripts/                   # thin CLI entry points calling into src/
├── results/                   # one JSON per run (committed)
├── reports/
│   ├── results.md             # generated (planned)
│   └── figures/
└── tests/
```

## 6. Implementation phases

### Phase 0 — Project setup
Directory layout, `pyproject.toml`, `.gitignore` (ignores `data/`, model checkpoints, `.venv`), `utils.py` with `set_seed(seed)` (random, numpy, torch), `get_device()` (cuda → mps → cpu) and a logger, and a `Makefile` with targets `setup`, `preprocess`, `rf`, `cnn`, `gru`, `pretrain`, `finetune`, `label-efficiency`, `ablation`, `report`, `test`.

### Phase 1 — Data preprocessing (the most important part)

**1.1 Session extraction (`pcap_reader.py`)**
- Read each pcap with `dpkt`, supporting pcap and pcapng.
- Keep only IPv4 packets carrying TCP or UDP; skip malformed packets (counted and logged).
- Bidirectional sessions: the key is a normalised 5-tuple `(min(ip:port), max(ip:port), proto)`, so both directions merge into one session.
- Per packet: timestamp, length, direction (+1 from the initiator, −1 otherwise), raw bytes starting at the IP header (Ethernet header removed).
- Filters: drop TCP packets without payload (pure ACKs) and DNS packets (port 53); drop sessions with fewer than `min_packets_per_session` packets.
- Limits: keep at most `max_packets_per_session` packets (default 32) and `max_sessions_per_class` sessions (default 5000, seeded random sampling).
- Streaming processing so large pcaps never fill the RAM (implemented as two passes: count, then collect sampled sessions).

**1.2 Three representations per session** (all models share the same sessions)
- **(a) Statistical features** (`features.py`, for Random Forest): packet count, total bytes, duration (per direction and overall); packet size min/max/mean/std/25th/50th/75th percentiles (per direction); inter-arrival time min/max/mean/std; packet and byte ratios between directions; protocol (TCP/UDP). IP and port are **not** used as features.
- **(b) Directed packet-size sequence** (for the GRU): vector of length `seq_len` (32) of `size × direction`, zero-padded, plus log-scaled inter-arrival time as a second channel.
- **(c) Byte representation** (`bytes_repr.py`, for CNN and Transformer): concatenate up to `bytes_per_packet` (64) bytes from each packet, cut/pad to `max_bytes` (784 for the CNN, following Wang et al. 2017; 512 tokens for the Transformer).
  - IP masking: zero the src/dst IP fields (`mask_ip`, default on).
  - Port masking: zero the src/dst ports (`mask_port`, default off; used for RQ3).
  - Also zero the IP checksum and the TCP/UDP checksum (they depend on the masked fields).
  - Stored as `uint8`, scaled to [0, 1] for the CNN.

**1.3 Dataset assembly and split (`build_dataset.py`)**
Outputs `sessions.parquet` (metadata + statistical features + labels), `seqs.npy`, `bytes/<variant>/{bytes,pkt_bytes,pkt_lens}.npy` for the mask variants `none`, `ip`, `ip_port`, `splits.json` and `meta.json`. Duplicates are removed by hashing the masked bytes before splitting, and a statistics table is printed (sessions per class before/after filtering, duplicates removed, train/val/test distribution).

**1.4 Tests** — a tiny pcap is generated with `dpkt` inside the tests to check that both directions merge into one session, directions are correct, IP bytes are masked at the right offsets, and padded length equals `max_bytes`.

### Phase 2 — Traditional ML baseline
`RandomForestClassifier` (`n_estimators=300`, `class_weight="balanced"`) and optionally `XGBClassifier` on features (a). Hyperparameters are chosen on the validation split; results are reported on test. Saves metrics, the confusion matrix and the top-15 feature importances, illustrating the interpretability advantage of traditional ML noted in the survey.

### Phase 3 — Deep learning
A shared training loop (`train_torch.py`): AdamW, cosine learning-rate schedule, early stopping on validation macro-F1 (patience 5), best-checkpoint saving, per-epoch logging, class weights for imbalanced data.
- **1D-CNN on bytes** (Wang et al. 2017, survey §4.2.2): 784 bytes → Conv1d(1→32, k=25) → ReLU → MaxPool → Conv1d(32→64, k=25) → ReLU → MaxPool → FC(1024) → Dropout(0.5) → FC(classes).
- **Bidirectional GRU on packet sizes** (simplified FS-Net, §4.2.3): 32 steps × 2 channels → 2-layer BiGRU, hidden 128 → concatenate the last hidden states of both directions → FC → classes.

### Phase 4 — Pre-trained Transformer (ET-BERT-style, scaled down)
- **Tokenisation**: one byte = one token (vocab 256) plus `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]` (260 total). `[SEP]` separates packets (similar in spirit to ET-BERT's burst structure); `[CLS]` starts the sequence; max length 512. ET-BERT uses byte bigrams (vocab 65,536); single bytes keep the model light enough for a laptop.
- **Architecture**: 4-layer Transformer encoder, `d_model=256`, 4 heads, FFN 1024, dropout 0.1. Embedding = token + learned position + packet-index embedding. Two heads: an MLM head for pre-training and a classification head on the `[CLS]` state for fine-tuning.
- **Pre-training** (`pretrain_mlm.py`): training split only, labels discarded. Masked Byte Modeling: select 15% of non-special tokens; of those, 80% → `[MASK]`, 10% → random byte, 10% unchanged (the BERT/ET-BERT rule, survey §5.3.1). Logs MLM loss and masked-token accuracy; saves the encoder checkpoint.
- **Fine-tuning** (`finetune.py`): load the pre-trained encoder, add the classification head, train everything with a small encoder learning rate (5e-5) and a larger head learning rate (1e-3). `--from-scratch` uses random initialisation (RQ2 baseline). `--label-fraction 0.01|0.05|0.1|1.0` takes a stratified, seeded subset of the labelled training split.

### Phase 5 — Evaluation and experiments
- **Metrics** (`metrics.py`, survey §2.4): accuracy and macro precision/recall/F1 (macro-F1 is the primary metric because of class imbalance), per-class F1, confusion matrix (image and numbers), parameter count, training time.
- **Latency** (`latency.py`): mean time per sample at batch size 1 and throughput at batch size 256, on CPU after warm-up — the survey's "real-time" criterion.
- Each run writes one JSON file to `results/`: model name, full config, seed, metrics, timings and git commit hash.
- **RQ1**: RF, CNN, GRU and the pre-trained Transformer (+ XGBoost if available) with 100% labels, seeds 42/43/44, reported as mean ± std.
- **RQ2**: `label_fraction ∈ {0.01, 0.05, 0.1, 1.0}` for the pre-trained Transformer, the from-scratch Transformer, the CNN and the RF; line plot with the label fraction on a log x-axis and macro-F1 on the y-axis (`make label-efficiency`).
- **RQ3**: CNN and Transformer with three settings — no masking, IP masked (default), IP + port masked. A large drop when masking indicates the model relies on network identifiers ("shortcut learning") rather than traffic characteristics (`make ablation`). All variants share the same `splits.json`.

### Phase 6 — Automated report and README
`report.py` reads all `results/*.json` and generates `reports/results.md` with the RQ1 table (model | accuracy | macro-F1 | parameters | training time | latency), the RQ2 figure (`reports/figures/label_efficiency.png`), the RQ3 table, the confusion matrix of the best model, and a "Limitations" section (old dataset with substantial plaintext; a single dataset; models scaled down from the originals; concept drift and open-set classification not studied). Optional: a Streamlit demo that classifies the sessions of an uploaded pcap.

## 7. Code conventions
- Full type hints; every module has a short docstring explaining its role and the relevant survey section.
- No hard-coded paths or hyperparameters in code — everything lives in `configs/*.yaml` and can be overridden from the CLI with `--set key.sub=value`.
- `set_seed` is called at the start of every script.
- Small, testable functions; I/O separated from logic.
- Comments and docstrings in English.
- One git commit per phase with a clear message (e.g. `phase1: session extraction and byte representation`).

## 8. Common pitfalls to avoid
1. Splitting by packet instead of by session → near-100% fake accuracy.
2. Pre-training on the full data including test → information leakage.
3. Masking IP addresses but not the checksums → the model can still infer the IP indirectly.
4. Using accuracy as the main metric on imbalanced data → use macro-F1.
5. Not removing duplicate sessions → the same sample appears in train and test.
6. Reading a whole pcap into RAM → process packets in a streaming fashion.
7. Byte order: read IP and port fields in network byte order (big-endian).
8. UDP has no handshake: the initiator is the sender of the first packet.

## 9. Decisions taken during implementation
- **Focus**: the pre-trained Transformer gets the in-depth treatment (pre-training, RQ2, RQ3, error analysis). Random Forest (ML generation) and 1D-CNN (DL generation) are the baselines; GRU and XGBoost are optional.
- **Minimum session length = 1 packet** (spec default was 3). In USTC-TFC2016 the benign classes BitTorrent, Facetime, MySQL, Outlook, Skype (and mostly Gmail, FTP) were generated with IXIA BreakingPoint and consist almost entirely of 1–2-packet sessions once pure ACKs are dropped; requiring ≥3 packets would delete them. **Tinba** is almost entirely DNS traffic and is excluded automatically (`min_sessions_per_class = 50`), leaving **19 classes**.
- All three mask variants are produced in one preprocessing run; de-duplication hashes the strictest variant (`ip_port`) so no variant has cross-split duplicates.
- A session is identified by its 5-tuple only (no idle timeout).
- Latency is measured on a single CPU thread for every model.
- The from-scratch Transformer uses a larger encoder learning rate (5e-4), since 5e-5 is a fine-tuning rate.
- With few labels, the number of epochs and the patience are stretched so that every torch model gets at least `min_steps` optimiser steps (applied identically to all models).

## 10. Quick start

```bash
make setup            # create .venv (python3.11) and install dependencies
make download         # fetch USTC-TFC2016 and extract it into data/raw/ustc-tfc2016/
make preprocess       # ~20 s on an Apple M2; quick run: make preprocess ARGS="--max-sessions-per-class 200"
make rf               # Random Forest with validation grid search
make cnn              # 1D-CNN
make gru              # BiGRU (optional)
make pretrain         # Masked Byte Modeling pre-training
make finetune         # fine-tuning; e.g. ARGS="--from-scratch --label-fraction 0.1"
make test             # unit tests
```

Any config value can be overridden with `--set key.sub=value` (e.g. `ARGS="--set train.epochs=5"`). Smoke runs with `--limit N` write to `results/_smoke/`, which is not committed.

## 11. Results

All phases are complete. Numbers below come from `results/*.json`; the generated report with figures is [`reports/results.md`](reports/results.md) and the detailed analysis (Vietnamese) is [`PROGRESS.md`](PROGRESS.md). Hardware: Apple M2, 16 GB RAM (training on MPS, latency on a single CPU thread).

**Dataset after preprocessing:** 83,930 sessions, 19 classes (Tinba excluded), train/val/test = 67,146 / 8,392 / 8,392.

### RQ1 — Generation comparison (100% labels, mean ± std over seeds 42/43/44)

| Model | Generation | Accuracy | **Macro-F1** | Size | Train time | Latency (batch 1) | Throughput (batch 256) |
|---|---|---|---|---|---|---|---|
| Random Forest | ML | 0.9301 ± 0.0002 | 0.9352 ± 0.0003 | 930,624 tree nodes | 3.4 s | 7.45 ms | ~24,950/s |
| BiGRU | DL | 0.8266 ± 0.0472 | 0.8289 ± 0.0468 | 402,707 params | 6.8 min | 1.57 ms | ~5,390/s |
| 1D-CNN | DL | 0.9880 ± 0.0013 | 0.9850 ± 0.0015 | 5,774,227 params | 6.4 min | 0.64 ms | ~3,510/s |
| Transformer, from scratch (seed 42 only) | DL | 0.9917 | 0.9908 | 3,504,151 params | 41.7 min | 15.29 ms | ~280/s |
| **Transformer, pre-trained** | Pre-training | 0.9917 ± 0.0013 | **0.9890 ± 0.0013** | 3,504,151 params | 41.8 min fine-tune + 42.3 min pre-train | 15.75 ms | ~281/s |

The Transformer is ~0.4–0.6 macro-F1 points better than the CNN, but costs ~13× more training time and ~25× more inference latency per sample.

### RQ2 — Label efficiency (seed 42, test macro-F1)

| Model | 1% labels | 5% | 10% | 100% |
|---|---|---|---|---|
| Random Forest | 0.8542 | 0.9084 | 0.9221 | 0.9351 |
| 1D-CNN | 0.8833 | 0.9569 | 0.9713 | 0.9835 |
| Transformer, from scratch | 0.9273 | 0.9659 | 0.9813 | 0.9908 |
| **Transformer, pre-trained** | **0.9476** | **0.9766** | 0.9794 | 0.9887 |

Pre-training helps only in the very-low-label regime (+0.020 at 1%, +0.011 at 5%); from 10% labels (~6,700 samples) on, the gain disappears. Most of the Transformer's advantage over the CNN comes from the architecture itself.

### RQ3 — IP / port masking (seed 42, test macro-F1; difference vs. the default IP-masked setting)

| Model | No masking | IP masked (default) | IP + port masked |
|---|---|---|---|
| 1D-CNN | 0.9839 (+0.0004) | 0.9835 | 0.9800 (−0.0035) |
| Transformer, pre-trained | 0.9909 (+0.0022) | 0.9887 | 0.9853 (−0.0034) |

IP addresses are not a shortcut on USTC-TFC2016; ports are a small one (the drop is concentrated on malware classes such as Shifu, Neris and Virut). With both masked the models still reach ~0.98, so the high scores mostly come from payload/header bytes — the dataset itself is easy, as the survey warns.

### Limitations
Old dataset with substantial plaintext and many IXIA-generated 1-packet sessions; a single dataset; scaled-down Transformer (byte tokens, ~3.5M parameters, 5 pre-training epochs — the MLM curve had flattened); RQ2/RQ3 use one seed; concept drift and open-set classification not studied.
