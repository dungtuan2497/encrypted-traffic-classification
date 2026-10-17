# Project: Phân loại lưu lượng mạng mã hóa — từ Machine Learning đến Pre-training

File này là bản đặc tả (spec) để xây dựng project từng bước. Làm lần lượt từng Phase, dừng lại sau mỗi Phase để báo cáo kết quả và chờ xác nhận trước khi sang Phase tiếp theo.

## 0. Các quyết định đã thống nhất (bổ sung so với spec gốc)

- **Trọng tâm**: Transformer pre-train + fine-tune được đầu tư sâu (pre-training, RQ2, RQ3, phân tích lỗi). Random Forest (đại diện học máy) và 1D-CNN (đại diện học sâu) là đối chứng. **GRU và XGBoost là tùy chọn**, chỉ làm nếu còn thời gian (chỉ góp mặt trong RQ1).
- **Môi trường**: `venv` + `requirements.txt` với `python3.11` (máy chưa có `uv`). XGBoost là phụ thuộc tùy chọn (`make setup-xgb`, macOS cần `brew install libomp`).
- **Vị trí**: project nằm ngay tại thư mục gốc repo (không tạo thư mục con `encrypted-traffic-classification/`).
- **Thứ tự**: Phase 0–1 → RF → CNN → Transformer → RQ2, RQ3, báo cáo → (GRU, XGBoost nếu còn thời gian).

## 1. Bối cảnh và mục tiêu

Project minh họa thực nghiệm cho bài báo:

> Dong et al., *Deep learning and pre-training technology for encrypted traffic classification: A comprehensive review*, Neurocomputing 617 (2025) 128444.

Bài báo chia các phương pháp phân loại lưu lượng mã hóa thành 4 thế hệ: luật plaintext → học máy truyền thống → học sâu → tiền huấn luyện (pre-training). Project cài đặt đại diện của 3 thế hệ sau trên cùng một bộ dữ liệu và cùng một cách chia dữ liệu, rồi so sánh chúng một cách công bằng.

### Câu hỏi nghiên cứu

1. **RQ1 — So sánh thế hệ**: Học máy truyền thống (Random Forest trên đặc trưng thống kê), học sâu (1D-CNN, GRU) và Transformer tiền huấn luyện khác nhau thế nào về độ chính xác, số tham số, thời gian huấn luyện và độ trễ suy luận?
2. **RQ2 — Hiệu quả dữ liệu nhãn**: Fine-tune với 1%, 5%, 10%, 100% nhãn và so với cùng kiến trúc huấn luyện từ đầu (from scratch).
3. **RQ3 — Học tắt (shortcut learning)**: Chạy lại thí nghiệm có và không che (mask) IP/port, xem độ chính xác thay đổi thế nào.

## 2. Phạm vi và giả định

- Chạy được trên laptop CPU; có GPU (CUDA hoặc Apple MPS) thì tự động dùng. Transformer dưới khoảng 5 triệu tham số.
- Mỗi bước tốn thời gian có tham số `--limit` hoặc `--max-sessions-per-class` để chạy thử nhanh.
- **Không bịa kết quả.** Mọi con số trong báo cáo phải sinh ra từ code đã chạy thật. Chưa chạy được thì ghi "chưa có kết quả".
- Thứ tự: chạy trên tập nhỏ → kiểm tra pipeline đúng → chạy tập đầy đủ.

## 3. Dữ liệu

### Bộ dữ liệu chính: USTC-TFC2016

20 lớp (10 lành tính, 10 mã độc):

- Lành tính: BitTorrent, Facetime, FTP, Gmail, MySQL, Outlook, Skype, SMB, Weibo, WorldOfWarcraft
- Mã độc: Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus
- Nguồn: `github.com/yungshenglu/USTC-TFC2016`. Đặt file `.pcap` (đã giải nén `.7z`/`.zip`) vào `data/raw/ustc-tfc2016/`, mỗi lớp một file hoặc một thư mục.
- Lưu ý cho báo cáo: bộ này có tỷ lệ plaintext đáng kể và flow dài, nên là bài toán tương đối dễ. Kết quả cao không đồng nghĩa với mô hình mạnh trên lưu lượng mã hóa thật.

### Bộ dữ liệu tùy chọn: ISCX VPN-nonVPN 2016

Code không phụ thuộc bộ dữ liệu: chỉ cần một file mapping `tên file pcap → nhãn`.

### Quy tắc bắt buộc chống rò rỉ dữ liệu

1. Chia theo session, không theo packet.
2. Loại trùng lặp trước khi chia (hash mảng byte đã tiền xử lý), ghi lại số lượng đã loại.
3. Pre-training chỉ dùng tập train (không nhãn), tuyệt đối không dùng val/test.
4. Chia một lần, lưu chỉ số vào `data/processed/splits.json`, mọi mô hình dùng chung.
5. Tỷ lệ 80/10/10, phân tầng theo nhãn, `seed=42`.

## 4. Tech stack

Python 3.11; `dpkt` (không dùng scapy cho xử lý hàng loạt); `numpy`, `pandas`, `pyarrow`; `scikit-learn`, `xgboost` (tùy chọn); `torch`; `pyyaml`; `argparse`; `matplotlib`; `tqdm`, `rich`; `pytest`.

## 5. Cấu trúc thư mục

```
├── configs/          default.yaml, rf.yaml, cnn.yaml, gru.yaml, transformer.yaml
├── data/raw/         pcap gốc (gitignore)
├── data/processed/   parquet/npy + splits.json (gitignore)
├── src/etc/
│   ├── config.py, utils.py
│   ├── data/         pcap_reader.py, features.py, bytes_repr.py, build_dataset.py, torch_datasets.py
│   ├── models/       rf.py, cnn1d.py, gru.py, transformer.py
│   ├── train/        train_sklearn.py, train_torch.py, pretrain_mlm.py, finetune.py
│   └── eval/         metrics.py, latency.py, report.py
├── scripts/          CLI mỏng, gọi vào src/
├── results/          JSON kết quả từng lần chạy (commit)
├── reports/          results.md (sinh tự động) + figures/
└── tests/
```

## 6. Các Phase triển khai

### Phase 0 — Khởi tạo project
Cấu trúc thư mục, `pyproject.toml`, `.gitignore`, `utils.py` (`set_seed`, `get_device` cuda→mps→cpu, logger), `Makefile` (setup, preprocess, rf, cnn, gru, pretrain, finetune, label-efficiency, ablation, report, test). Hoàn thành khi `make setup && make test` chạy được.

### Phase 1 — Tiền xử lý dữ liệu
**1.1 Tách session**: dpkt, pcap + pcapng; chỉ IPv4 TCP/UDP, đếm gói lỗi; session hai chiều với khóa `(min(ip:port), max(ip:port), proto)`; mỗi packet lưu timestamp, độ dài, hướng (+1 phía khởi tạo, −1 ngược lại), byte thô từ IP header. Lọc: session < 3 gói, gói TCP không payload, DNS (port 53). Giới hạn `max_packets_per_session` (32), `max_sessions_per_class` (5000, lấy mẫu có seed). Streaming.

**1.2 Ba dạng biểu diễn**:
- (a) Đặc trưng thống kê cho RF: số gói, byte, thời lượng (từng chiều và tổng); kích thước gói min/max/mean/std/p25/p50/p75 (từng chiều); IAT min/max/mean/std; tỷ lệ gói và byte giữa hai chiều; giao thức. Không dùng IP/port.
- (b) Chuỗi `kích_thước × hướng` dài 32 (pad 0) + kênh IAT log-scale, cho GRU.
- (c) Byte: tối đa 64 byte/gói, nối lại, cắt/pad về `max_bytes` (784 cho CNN, 512 cho Transformer). Che IP (`mask_ip`, mặc định bật), che port (`mask_port`, mặc định tắt), che IP checksum và TCP/UDP checksum. Lưu `uint8`.

**1.3 Ghép dataset**: `sessions.parquet`, `bytes.npy`, `seqs.npy`, `splits.json`. Loại trùng theo hash byte sau mask rồi mới chia. In bảng thống kê.

**1.4 Kiểm thử**: pcap nhỏ tạo bằng dpkt trong test: gộp hai chiều đúng, hướng đúng, che IP đúng vị trí, độ dài pad đúng.

Hoàn thành khi `make preprocess` chạy được với `--max-sessions-per-class 200`, in bảng thống kê, test pass.

### Phase 2 — Baseline học máy
RandomForest (`n_estimators=300`, `class_weight="balanced"`) và XGBoost (tùy chọn). Chọn siêu tham số trên val, báo cáo trên test. Lưu metrics, ma trận nhầm lẫn, top 15 feature importance.

### Phase 3 — Học sâu
Vòng lặp chung: AdamW, cosine LR, early stopping theo macro-F1 val (patience 5), checkpoint tốt nhất, log mỗi epoch, class weight.
- 1D-CNN: 784 byte → Conv1d(1→32,k25) → ReLU → MaxPool → Conv1d(32→64,k25) → ReLU → MaxPool → FC(1024) → Dropout(0.5) → FC(số lớp).
- BiGRU (tùy chọn): 32 bước × 2 kênh → BiGRU 2 lớp hidden 128 → nối hidden cuối hai chiều → FC.

### Phase 4 — Transformer tiền huấn luyện
- Token: byte (256) + `[PAD] [CLS] [SEP] [MASK]` = 260; `[SEP]` giữa các gói; `[CLS]` ở đầu; tối đa 512. (ET-BERT dùng bigram, vocab 65.536.)
- Kiến trúc: 4 lớp, d_model 256, 4 head, FFN 1024, dropout 0.1; token + position + packet-index embedding; MLM head và classification head (`[CLS]`). In số tham số.
- Pre-training: chỉ tập train; 15% token (không phải token đặc biệt), 80% `[MASK]` / 10% ngẫu nhiên / 10% giữ nguyên; 10–20 epoch; log MLM loss và accuracy.
- Fine-tuning: lr 5e-5 encoder, 1e-3 head; `--from-scratch`; `--label-fraction 0.01|0.05|0.1|1.0` (lấy mẫu phân tầng, seed cố định).

### Phase 5 — Đánh giá và thí nghiệm
- Metrics: accuracy, macro P/R/F1 (chính), F1 từng lớp, ma trận nhầm lẫn, số tham số, thời gian train, latency batch=1 và throughput batch=256 trên CPU sau warm-up. Mỗi lần chạy lưu JSON vào `results/` (tên mô hình, config, seed, metrics, thời gian, git commit).
- RQ1: RF, CNN, Transformer (+ XGBoost, GRU nếu có) với 100% nhãn, seed 42/43/44, trung bình ± độ lệch chuẩn.
- RQ2: label_fraction ∈ {0.01, 0.05, 0.1, 1.0} cho Transformer pre-trained, from scratch, CNN, RF; biểu đồ x log-scale, y macro-F1.
- RQ3: CNN và Transformer với (1) không che, (2) che IP, (3) che IP + port; mỗi cấu hình thư mục con riêng, dùng chung `splits.json`.

### Phase 6 — Báo cáo và README
`report.py` sinh `reports/results.md` (bảng RQ1, biểu đồ RQ2, bảng RQ3, ma trận nhầm lẫn mô hình tốt nhất, mục Hạn chế). README tiếng Việt. (Tùy chọn) demo Streamlit.

## 7. Quy ước code
Type hint đầy đủ; docstring module nêu liên hệ với bài báo; không hard-code đường dẫn/siêu tham số (dùng `configs/*.yaml`, CLI ghi đè); `set_seed` đầu mọi script; tách I/O khỏi logic; comment/docstring tiếng Anh, báo cáo và README tiếng Việt; commit git sau mỗi Phase.

## 8. Lỗi thường gặp cần tránh
1. Chia theo packet thay vì session. 2. Pre-train trên cả test. 3. Che IP mà quên checksum. 4. Dùng accuracy làm chỉ số chính. 5. Không loại session trùng. 6. Đọc toàn bộ pcap vào RAM. 7. Sai byte order (network byte order, big-endian). 8. UDP: phía khởi tạo là phía gửi gói đầu tiên.

## 9. Hướng dẫn làm việc
Làm từng Phase; cuối Phase chạy test, chạy thử tập nhỏ, báo cáo số liệu thật, chờ đồng ý. Thiếu dữ liệu thì dừng và nhắc tải (không tạo dữ liệu giả ngoài unit test). Lựa chọn chưa đặc tả: chọn phương án đơn giản nhất, ghi chú lý do. Ước tính > 1 giờ/lần chạy trên CPU thì đề xuất cách giảm trước. Không ghi số liệu vào báo cáo nếu không đến từ `results/`.
