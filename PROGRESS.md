# Phân loại lưu lượng mạng mã hóa — từ Machine Learning đến Pre-training

> Tài liệu tổng hợp tiến độ project (cập nhật 17/10/2026).
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
| Học máy truyền thống | Random Forest | 54 đặc trưng thống kê của session |
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
| 2 | Random Forest: chọn siêu tham số trên val, đánh giá trên test, feature importance | ✅ Xong (1 seed) |
| 3 | Vòng lặp huấn luyện chung (AdamW, cosine LR, early stopping theo macro-F1, class weight); 1D-CNN; BiGRU | ✅ Xong (1 seed) |
| 4 | Transformer: tokenizer, mô hình, pre-train MLM, fine-tune (`--from-scratch`, `--label-fraction`) | 🟡 **Code xong, chưa chạy đầy đủ** |

Kiểm thử: **26/26 test pass** (`make test`). Có test dùng pcap tự tạo để kiểm tra gộp session hai chiều, hướng gói, vị trí byte bị che, độ dài pad, quy tắc che 80/10/10 của BERT và việc không rò rỉ giữa các split.

### Kết quả hiện có (tập test, seed 42, 19 lớp)

| Mô hình | Thế hệ | Accuracy | **Macro-F1** | Kích thước | Thời gian train | Độ trễ (batch=1, 1 luồng CPU) | Throughput (batch=256) |
|---|---|---|---|---|---|---|---|
| Random Forest | Học máy | 0,9301 | 0,9351 | 930.624 nút cây | 3,6 giây | 7,56 ms | ~24.840 mẫu/giây |
| **1D-CNN** (che IP) | Học sâu | **0,9864** | **0,9835** | 5.774.227 tham số | 315 giây | 0,73 ms | ~2.520 mẫu/giây |
| BiGRU | Học sâu | 0,7880 | 0,7916 | 402.707 tham số | 265 giây | 1,57 ms | ~5.260 mẫu/giây |
| Transformer pre-trained | Pre-training | *chưa có kết quả* | *chưa có kết quả* | 3.504.151 tham số | — | — | — |

Phần cứng: MacBook Apple M2, RAM 16 GB. Train trên GPU Apple (MPS), đo độ trễ trên CPU.

**Nhận xét ban đầu:**
- **Random Forest** nhầm nhiều nhất ở BitTorrent và Outlook (36% BitTorrent bị đoán thành Outlook; cả hai đều là session 1 gói) và ở SMB và Weibo (nhầm qua lại 21–26%). Đặc trưng quan trọng nhất là kích thước gói chiều đi, giao thức TCP/UDP và khoảng thời gian giữa các gói. Phần này minh họa ưu điểm "dễ giải thích" của học máy.
- **1D-CNN** đọc được byte nội dung nên sửa gần hết các lỗi trên: SMB, Weibo, BitTorrent, Outlook đều đạt F1 > 0,99. Lớp yếu nhất còn lại là Virut (0,87) và Neris (0,90), đều là mã độc.
- **BiGRU** chỉ thấy kích thước và thời gian gói, nên gần như "mù" với các lớp có session 1 gói (SMB 0,33; BitTorrent 0,54). Mô hình chưa được tinh chỉnh vì là mô hình tùy chọn.
- **Độ trễ**: CNN nhanh hơn RF khi xử lý từng mẫu, nhưng RF có throughput cao hơn khoảng 10 lần khi xử lý theo lô.

Kết quả chi tiết (F1 từng lớp, ma trận nhầm lẫn, lịch sử huấn luyện) nằm trong `results/*.json`. Hình nằm trong `reports/figures/`.

---

## 4. Chưa làm được / đang dở

| Việc | Ghi chú |
|---|---|
| **Pre-train Transformer trên dữ liệu đầy đủ** | Lần chạy thử cho thấy MLM loss giảm (pipeline đúng), nhưng tốc độ trên M2 ước tính khoảng **25 phút/epoch**, nên 10 epoch hết hơn 4 giờ. Đã sửa (làm tròn độ dài batch lên bội số 64 để GPU Apple không phải biên dịch lại liên tục) nhưng **chưa đo lại**. Nếu vẫn chậm, sẽ đề xuất giảm số epoch hoặc pre-train trên một phần tập train. |
| Fine-tune Transformer (100% nhãn) | Chờ pre-train. |
| **RQ1** — 3 seed (42, 43, 44), trung bình ± độ lệch chuẩn | Mới có seed 42 cho RF, CNN, GRU. |
| **RQ2** — 1%, 5%, 10%, 100% nhãn | Code đã hỗ trợ (`--label-fraction`, lấy mẫu phân tầng, `min_steps` để công bằng khi ít nhãn). Chưa chạy. |
| **RQ3** — che: không / IP / IP + port | Dữ liệu ba biến thể đã sinh sẵn trong `data/processed/bytes/{none,ip,ip_port}/`, CLI có `--mask`. Chưa chạy. |
| Báo cáo tự động `reports/results.md` + biểu đồ RQ2 + bảng RQ3 | Chưa viết `report.py`. |
| README đầy đủ (tiếng Việt) | Chưa. |
| XGBoost (tùy chọn) | Code có sẵn (`make xgb`), cần `brew install libomp`. |
| Demo Streamlit (tùy chọn) | Chưa. |

### Các điều chỉnh so với đặc tả ban đầu (đều có ghi chú trong code)
- `min_packets_per_session = 1` thay vì 3 (phương án A ở trên); lớp nào có dưới 50 session thì bị loại tự động.
- Sinh luôn cả 3 biến thể che trong một lần tiền xử lý. Loại trùng dùng hash của biến thể che nặng nhất (`ip_port`), nên không biến thể nào bị trùng giữa train và test.
- Session xác định bằng 5-tuple, không có timeout.
- Độ trễ đo trên **1 luồng CPU** cho mọi mô hình để so sánh công bằng.
- Transformer from-scratch dùng learning rate encoder lớn hơn (5e-4 thay vì 5e-5), vì 5e-5 là mức dành cho fine-tune.

---

## 5. Cách chạy

```bash
make setup            # tạo .venv (Python 3.11) và cài thư viện
make download         # tải USTC-TFC2016 và giải nén vào data/raw/ustc-tfc2016/
make preprocess       # khoảng 20 giây trên M2; chạy thử: make preprocess ARGS="--max-sessions-per-class 200"
make rf               # Random Forest (khoảng 1 phút gồm grid search)
make cnn              # 1D-CNN (khoảng 5 phút trên M2)
make gru              # BiGRU (khoảng 4–5 phút)
make pretrain         # Transformer MLM (đang tối ưu tốc độ)
make finetune         # Fine-tune; thêm ARGS="--from-scratch --label-fraction 0.1" ...
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
