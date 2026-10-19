# Phân loại lưu lượng mạng mã hóa — từ Machine Learning đến Pre-training

> Tài liệu tổng hợp tiến độ project (cập nhật 19/10/2026 — đã hoàn thành Phase 0–6).
> Mọi con số trong file này lấy từ các file kết quả thật trong `results/` và `data/processed/meta.json`. Mục nào chưa chạy được ghi rõ **"chưa có kết quả"**.

---

## 1. Project này làm gì?

Hiện nay phần lớn lưu lượng Internet đã được **mã hóa** (HTTPS, TLS, VPN), nên không thể đọc nội dung gói tin. Dù vậy, mỗi ứng dụng vẫn để lại "dấu vân tay" qua kích thước gói, nhịp độ gửi, hướng đi/về và vài byte header chưa bị mã hóa.

Project xây dựng và **so sánh công bằng** các mô hình AI đoán một *session* mạng (một cuộc trao đổi giữa hai máy) thuộc ứng dụng nào (Skype, Gmail, FTP…) hoặc loại **mã độc** nào (Zeus, Cridex, Neris…), **mà không cần giải mã**.

Đây là phần thực nghiệm minh họa cho bài báo tổng quan:

> Dong et al., *Deep learning and pre-training technology for encrypted traffic classification: A comprehensive review*, Neurocomputing 617 (2025) 128444.

Bài báo chia lịch sử lĩnh vực thành 4 thế hệ: luật plaintext → **học máy** → **học sâu** → **tiền huấn luyện (pre-training)**. Project cài đặt đại diện của 3 thế hệ sau trên **cùng dữ liệu, cùng cách chia train/val/test**:

| Thế hệ | Mô hình | Đầu vào |
|---|---|---|
| Học máy truyền thống | Random Forest | 45 đặc trưng thống kê của session |
| Học sâu | 1D-CNN (Wang et al. 2017) | 784 byte đầu của session |
| Học sâu (tùy chọn) | BiGRU (rút gọn từ FS-Net) | Chuỗi kích thước gói + thời gian giữa các gói |
| **Pre-training** (trọng tâm) | Transformer kiểu ET-BERT, khoảng 3,5 triệu tham số | Chuỗi token byte, pre-train bằng Masked Byte Modeling |

### Liên hệ với môn "Các xu hướng trong chuyển đổi số"

1. **Tiến hóa của AI**: từ con người tự thiết kế đặc trưng, sang máy tự học đặc trưng, rồi đến *foundation model* (học trước trên dữ liệu không nhãn, cùng tư tưởng với BERT/GPT). Project cho thấy xu hướng này lan sang cả lĩnh vực an ninh mạng.
2. **An ninh mạng trong chuyển đổi số**: mã hóa bảo vệ quyền riêng tư nhưng cũng giúp mã độc ẩn mình. Phát hiện mã độc *không cần giải mã* giải quyết được mâu thuẫn này.
3. **Chi phí dữ liệu nhãn**: doanh nghiệp có nhiều dữ liệu nhưng ít nhãn. RQ2 kiểm tra pre-training có giảm được phụ thuộc vào nhãn hay không.
4. **AI đáng tin cậy**: RQ3 kiểm tra các con số "99%" có phải do mô hình *học tắt* (nhớ địa chỉ IP/port) hay không. Đây là góc nhìn phản biện đối với xu hướng.

### Ba câu hỏi nghiên cứu

- **RQ1 — So sánh thế hệ**: độ chính xác, số tham số, thời gian huấn luyện, độ trễ suy luận.
- **RQ2 — Hiệu quả dữ liệu nhãn**: fine-tune với 1%, 5%, 10%, 100% nhãn; so sánh pre-trained với from-scratch.
- **RQ3 — Học tắt**: chạy lại khi không che, che IP, che IP + port.

---

## 2. Dữ liệu: USTC-TFC2016

- Nguồn: <https://github.com/yungshenglu/USTC-TFC2016>. Lệnh `make download` tự tải và giải nén các file `.7z`, tổng 3,8 GB pcap.
- 20 lớp gốc: 10 lành tính, 10 mã độc.
- Các quy tắc chống rò rỉ dữ liệu đã cài đặt:
  - chia theo **session**, không theo gói;
  - **loại session trùng** (hash byte sau khi che) *trước khi* chia;
  - chia **một lần** theo tỷ lệ 80/10/10, phân tầng theo lớp, `seed=42`, lưu vào `data/processed/splits.json` để mọi mô hình dùng chung;
  - pre-training **chỉ dùng tập train**;
  - che IP, port, IP checksum và TCP/UDP checksum (checksum phụ thuộc IP nên cũng phải che).

### Phát hiện quan trọng về chất lượng dữ liệu

Khi tách session, thấy rằng phần lưu lượng lành tính của bộ dữ liệu được **sinh bằng công cụ giả lập IXIA BreakingPoint**: địa chỉ dải `1.1.x.x → 1.2.x.x`, mỗi luồng rất ngắn. Sau khi bỏ gói ACK rỗng và DNS:

| Kiểu session | Lớp |
|---|---|
| 100% session chỉ có **1 gói** | BitTorrent, Facetime, MySQL, Outlook, Skype |
| Phần lớn 1 gói | FTP (70%), Gmail (80%) (đo trên tập chạy thử 200 session/lớp) |
| Session dài (trung vị 32 gói) | Weibo, SMB |
| Gần như toàn bộ là **DNS** | Tinba, chỉ còn 1 session không phải DNS nên **bị loại** |

**Quyết định (phương án A):** hạ điều kiện tối thiểu từ ≥3 gói xuống **≥1 gói có payload**, vẫn bỏ DNS và ACK rỗng. Kết quả còn **19 lớp**.
**Ý nghĩa cho báo cáo:** với khoảng 7 lớp, bài toán thực chất là phân loại *từng gói tin đơn lẻ*. Vì vậy cần nhìn các con số "99%" trên bộ này một cách thận trọng.

### Dataset sau xử lý (chạy đầy đủ, tối đa 5.000 session/lớp)

| | Giá trị |
|---|---|
| Gói đã đọc | 5.787.440 |
| Session sau lọc và lấy mẫu | 83.932 |
| Session trùng đã loại | 1 |
| **Session cuối cùng / số lớp** | **83.930 / 19** |
| Train / Val / Test | 67.146 / 8.392 / 8.392 |

Các lớp ít dữ liệu: Shifu 362, Zeus 504, Virut 3.461, Htbot 4.700, Miuref 4.903 session.

---

## 3. Đã làm được

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khung project, cấu hình YAML, Makefile, tiện ích (seed, device, logger) | ✅ Xong |
| 1 | Đọc pcap theo kiểu streaming (dpkt), tách session hai chiều, 3 dạng biểu diễn, 3 biến thể che IP/port, loại trùng, chia split | ✅ Xong |
| 2 | Random Forest: chọn siêu tham số trên val, đánh giá trên test, feature importance | ✅ Xong |
| 3 | Vòng lặp huấn luyện chung; 1D-CNN; BiGRU | ✅ Xong |
| 4 | Transformer: tokenizer, mô hình 3,5 triệu tham số, pre-train MLM (chỉ dùng tập train), fine-tune | ✅ Xong |
| 5 | RQ1 (3 seed), RQ2 (1/5/10/100% nhãn), RQ3 (3 kiểu che) — 32 lần chạy (29 lần huấn luyện phân loại + 3 lần pre-train), không có lần nào lỗi | ✅ Xong |
| 6 | Báo cáo tự động `reports/results.md` + biểu đồ (`make report`) | ✅ Xong |

Kiểm thử: **28/28 test pass** (`make test`).
Toàn bộ số liệu chi tiết (F1 từng lớp, ma trận nhầm lẫn, lịch sử huấn luyện) nằm trong `results/*.json`. Bảng tổng hợp và biểu đồ nằm trong [`reports/results.md`](reports/results.md).
Phần cứng: MacBook Apple M2, RAM 16 GB. Train trên GPU Apple (MPS), đo độ trễ trên **1 luồng CPU**.

### 3.1. RQ1 — So sánh các thế hệ (100% nhãn, trung bình ± độ lệch chuẩn qua seed 42/43/44)

| Mô hình | Thế hệ | Accuracy | **Macro-F1** | Kích thước | Thời gian train | Độ trễ / mẫu | Throughput (batch 256) |
|---|---|---|---|---|---|---|---|
| Random Forest | Học máy | 0,9301 ± 0,0002 | 0,9352 ± 0,0003 | 930.624 nút cây | 3,4 giây | 7,45 ms | ~24.950 mẫu/giây |
| BiGRU | Học sâu | 0,8266 ± 0,0472 | 0,8289 ± 0,0468 | 402.707 tham số | 6,8 phút | 1,57 ms | ~5.390 mẫu/giây |
| 1D-CNN | Học sâu | 0,9880 ± 0,0013 | 0,9850 ± 0,0015 | 5.774.227 tham số | 6,4 phút | **0,64 ms** | ~3.510 mẫu/giây |
| Transformer học từ đầu (1 seed) | Học sâu | 0,9917 | **0,9908** | 3.504.151 tham số | 41,7 phút | 15,29 ms | ~280 mẫu/giây |
| **Transformer có pre-train** | Pre-training | **0,9917 ± 0,0013** | **0,9890 ± 0,0013** | 3.504.151 tham số | 41,8 phút fine-tune + 42,3 phút pre-train | 15,75 ms | ~281 mẫu/giây |

**Nhận xét:**
- Xếp hạng độ chính xác: **Transformer ≈ CNN > RF ≫ GRU**. Transformer (có hay không pre-train) hơn CNN khoảng 0,4–0,6 điểm macro-F1, lớn hơn độ lệch chuẩn giữa các seed (~0,0015) nên là khác biệt thật, nhưng nhỏ.
- **Cái giá** của Transformer: tổng thời gian train gấp khoảng **13 lần** CNN (khoảng 84 phút so với 6,4 phút). Khi suy luận, Transformer chậm hơn CNN khoảng **25 lần** mỗi mẫu, và throughput chỉ bằng khoảng **1/90** so với RF. Đây chính là đánh đổi giữa độ chính xác và khả năng chạy thời gian thực mà bài báo nêu.
- **Random Forest** train trong 3 giây, ổn định nhất (độ lệch chuẩn 0,0003) và dễ giải thích (feature importance), nhưng kém 5 điểm vì không đọc được nội dung byte. RF nhầm nhiều ở các lớp có session 1 gói (BitTorrent ↔ Outlook) và giữa SMB ↔ Weibo.
- **GRU** yếu và không ổn định (0,79 → 0,88 tùy seed), vì nó chỉ nhìn kích thước và thời gian của gói, mà nhiều lớp chỉ có 1 gói. Mô hình chưa được tinh chỉnh vì là mô hình tùy chọn.
- Lớp khó nhất với mọi mô hình học sâu là **Virut** và **Neris** (F1 khoảng 0,87–0,93), đều là mã độc.

### 3.2. RQ2 — Hiệu quả dữ liệu nhãn (seed 42, macro-F1 trên tập test)

| Mô hình | 1% nhãn (672 mẫu) | 5% | 10% | 100% |
|---|---|---|---|---|
| Random Forest | 0,8542 | 0,9084 | 0,9221 | 0,9351 |
| 1D-CNN | 0,8833 | 0,9569 | 0,9713 | 0,9835 |
| Transformer học từ đầu | 0,9273 | 0,9659 | 0,9813 | 0,9908 |
| **Transformer có pre-train** | **0,9476** | **0,9766** | 0,9794 | 0,9887 |
| *Lợi ích của pre-train* | *+0,020* | *+0,011* | *−0,002* | *−0,002* |

Biểu đồ: [`reports/figures/label_efficiency.png`](reports/figures/label_efficiency.png).

**Nhận xét:**
- Bài báo cho rằng pre-train giúp giảm phụ thuộc vào dữ liệu có nhãn. Kết quả **ủng hộ có điều kiện**: pre-train chỉ giúp rõ khi **rất ít nhãn** (1% nhãn: +2,0 điểm; 5% nhãn: +1,1 điểm). Từ 10% nhãn (khoảng 6.700 mẫu) trở lên, lợi ích **biến mất**; chênh lệch 0,002 nằm trong dao động giữa các lần chạy.
- **Kiến trúc Transformer** đóng góp nhiều hơn bản thân pre-train. Ở 1% nhãn, Transformer học từ đầu đã hơn CNN 4,4 điểm, còn pre-train chỉ thêm 2 điểm.
- Với chỉ 5% nhãn, Transformer có pre-train (0,9766) đã gần bằng CNN dùng 100% nhãn (0,9835).
- Có khả năng lợi ích của pre-train sẽ lớn hơn trên bộ dữ liệu khó hơn, hoặc khi pre-train trên lượng dữ liệu không nhãn lớn hơn nhiều so với dữ liệu fine-tune.

### 3.3. RQ3 — Học tắt: che IP / port (seed 42, macro-F1 trên tập test)

| Mô hình | Không che | Che IP (mặc định) | Che IP + port |
|---|---|---|---|
| 1D-CNN | 0,9839 (+0,0004) | 0,9835 | 0,9800 (−0,0035) |
| Transformer có pre-train | 0,9909 (+0,0022) | 0,9887 | 0,9853 (−0,0034) |

**Nhận xét:**
- **Địa chỉ IP không phải lối tắt** trên USTC-TFC2016: bỏ che IP chỉ thay đổi dưới 0,25 điểm. Một lý do là phần lưu lượng lành tính do IXIA sinh ra dùng IP ngẫu nhiên ở mỗi luồng.
- **Port là một lối tắt nhỏ**: che port làm giảm khoảng 0,35 điểm, chủ yếu ở các lớp **mã độc** (Transformer: Shifu −0,039, Neris −0,023, Virut −0,019; CNN: Neris −0,019, Virut −0,014). Mã độc thường dùng các cổng đặc trưng.
- Khi che cả IP lẫn port, mô hình vẫn đạt khoảng **0,98**. Nghĩa là các con số 98–99% trên bộ này **chủ yếu đến từ nội dung byte** (bộ dữ liệu có nhiều plaintext và session 1 gói dễ nhận dạng), không phải do mô hình "ăn gian" bằng định danh mạng. Bài toán trên USTC-TFC2016 vốn **dễ**, đúng như bài báo đã cảnh báo.

### 3.4. Pre-training (Masked Byte Modeling)

| Biến thể mask | MLM loss (held-out) | Độ chính xác byte bị che | Thời gian |
|---|---|---|---|
| ip (mặc định) | 5,707 → 1,766 | 0,3% → 67,6% | 42,3 phút |
| none | 5,732 → 2,325 | 0,3% → 57,8% | 42,5 phút |
| ip_port | 5,702 → 1,595 | 0,2% → 70,8% | 42,0 phút |

Đường cong loss đi ngang ở epoch 4–5 (epoch 5 chỉ giảm 0,007), nên 5 epoch là đủ. Loss của biến thể `none` cao hơn vì byte IP gần như ngẫu nhiên, không đoán được; hai con số MLM giữa các biến thể không so sánh trực tiếp được.

---

## 4. Chưa làm / hạn chế

| Việc | Ghi chú |
|---|---|
| RQ2, RQ3 nhiều seed | Hiện chỉ chạy seed 42. Các chênh lệch nhỏ (< 0,005) nên được xem là ngang nhau. |
| Pre-train 10 epoch như spec | Rút gọn còn 5 epoch (phương án 1) do phần cứng; đường cong đã đi ngang nên ảnh hưởng nhỏ. |
| Transformer học từ đầu với 3 seed | Mới có seed 42 (đây là baseline của RQ2). |
| Bộ dữ liệu thứ hai (ISCX VPN-nonVPN) | Chưa làm; code đã hỗ trợ đổi bộ dữ liệu qua file mapping. |
| XGBoost, demo Streamlit (tùy chọn) | Chưa làm. XGBoost có sẵn code (`make xgb`, cần `brew install libomp`). |
| Concept drift, tập mở, tấn công đối kháng | Ngoài phạm vi. |

### Các điều chỉnh so với đặc tả ban đầu (đều có ghi chú trong code)
- `min_packets_per_session = 1` thay vì 3 (phương án A); lớp có dưới 50 session bị loại tự động (Tinba), còn 19 lớp.
- Sinh luôn cả 3 biến thể che trong một lần tiền xử lý. Loại trùng dùng hash của biến thể che nặng nhất (`ip_port`), nên không biến thể nào bị trùng giữa train và test.
- Session xác định bằng 5-tuple, không có timeout.
- Độ trễ đo trên **1 luồng CPU** cho mọi mô hình.
- Transformer: pre-train 5 epoch, fine-tune tối đa 5 epoch, batch 32 (do giới hạn bộ nhớ của M2 16 GB). Fine-tune ở seed 43/44 dùng chung encoder pre-train seed 42.
- Transformer from-scratch dùng learning rate encoder lớn hơn (5e-4 thay vì 5e-5), vì 5e-5 là mức dành cho fine-tune.
- Khi ít nhãn, số epoch được giãn để mọi mô hình torch có ít nhất 1.500 bước tối ưu, với số lần đánh giá trên val giữ nguyên (áp dụng như nhau cho mọi mô hình).

---

## 5. Cách chạy

```bash
make setup            # tạo .venv (Python 3.11) và cài thư viện
make download         # tải USTC-TFC2016 và giải nén vào data/raw/ustc-tfc2016/
make preprocess       # khoảng 20 giây trên M2; chạy thử: make preprocess ARGS="--max-sessions-per-class 200"
make rf               # Random Forest (khoảng 1 phút gồm grid search)
make cnn              # 1D-CNN (khoảng 5 phút trên M2)
make gru              # BiGRU (khoảng 4–5 phút)
make pretrain         # Transformer MLM (~42 phút trên M2)
make finetune         # Fine-tune (~42 phút); thêm ARGS="--from-scratch --label-fraction 0.1" ...
make experiments      # toàn bộ RQ1–RQ3 (~6,5 giờ trên M2, chạy nền được)
make report           # sinh reports/results.md
make test             # chạy toàn bộ unit test
```

Có thể ghi đè mọi tham số: `--set khoa.con=gia_tri`. Chạy thử nhanh dùng `--limit N` (kết quả lưu vào `results/_smoke/`, không commit).

## 6. Cấu trúc thư mục

```
configs/              cấu hình YAML (default, rf, cnn, gru, transformer, datasets/)
src/etc/data/         đọc pcap, đặc trưng, biểu diễn byte, ghép dataset, loader
src/etc/models/       rf, cnn1d, gru, transformer
src/etc/train/        train_sklearn, train_torch (vòng lặp chung), pretrain_mlm, finetune
src/etc/eval/         metrics, latency
scripts/              CLI mỏng gọi vào src/
results/              JSON kết quả từng lần chạy (có commit)
reports/figures/      ma trận nhầm lẫn, đường cong học, feature importance
tests/                unit test (pytest)
PROJECT_GUIDE.md      đặc tả chi tiết của project
```

Dữ liệu (`data/`) và checkpoint mô hình (`checkpoints/`) **không** được đưa lên git.
