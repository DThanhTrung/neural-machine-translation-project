# neural-machine-translation-project

Implementation of seq2seq attention (LSTM) and Transformers from scratch to translate from English to Vietnamese.

Cấu trúc tổng thể dự kiến của dự án được mô tả dưới đây. Nhánh
`huy-dataset-loader` đã triển khai pipeline loader trong `src/data/`, script
preview và kiểm thử; tokenizer thật, cấu hình và các mô hình là phần tích hợp
tiếp theo của nhóm, chưa có đầy đủ trên nhánh này.

```
neural-machine-translation-project/
├── configs/                  # File cấu hình tham số (hyperparameters)
│   ├── lstm_config.yaml
│   └── transformer_config.yaml
│   └── general_config.yaml
├── data/                     # Thư mục chứa dữ liệu
│   ├── raw/                  # Dữ liệu gốc (vi-en)
│   └── processed/            # Dữ liệu đã tokenize, lưu dạng tensor/pickle
├── src/                      # Mã nguồn chính
│   ├── __init__.py
│   ├── data/                 # Xử lý dữ liệu & Tokenizer
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # Dự kiến: tokenizer thật do phần tokenizer cung cấp
│   │   ├── tokenizer_protocol.py # Giao diện tokenizer để loader sử dụng
│   │   ├── validation.py     # Kiểm tra vocabulary, special token và ID
│   │   ├── readers.py        # Đọc và kiểm tra cặp câu Anh–Việt
│   │   ├── dataset.py        # Encode và chuẩn bị từng mẫu
│   │   ├── batch.py          # Cấu trúc sample và batch mở rộng
│   │   ├── collator.py       # Gom mẫu, padding và lấy độ dài
│   │   └── loader.py         # API tạo PyTorch DataLoader
│   ├── models/               # Định nghĩa các mô hình mạng
│   │   ├── __init__.py
│   │   ├── lstm.py		        # Seq2Seq LSTM (Encoder, Decoder, Attention)
│   │   └── transformer.py    # Transformer (Multi-Head Attention, Positional Encoding, etc.)
│   ├── modules/              # Các module phụ trợ
│   │   ├── __init__.py
│   │   ├── beam_search.py    # Thuật toán Beam Search cho inference
│   │   └── embedding.py      # Thuật toán Embedding từ trước khi đưa vào mô hình
│   │   ├── lstm/             # Các module phụ trợ của riêng mô hình LSTM (Encoder, Decoder, Attention của LSTM, …)
│   │   ├── transformer/      # Các module phụ trợ của riêng mô hình Transformers (Positional Embedding, Self Attention, Cross Attention, FFN, Layer Norm, Activation, …)
│   ├── training/             # Vòng lặp huấn luyện & tối ưu
│   │   ├── __init__.py
│   │   ├── trainer.py        # Class Trainer tổng quát (train, validation, checkpoint)
│   │   └── optimizer.py      # Optimizer & Learning Rate Scheduler (vd: Noam Scheduler)
│   └── evaluation/           # Đánh giá & Trực quan hóa
│       ├── __init__.py
│       ├── metrics.py        # Tính toán BLEU score (SacreBLEU / NLTK)
│       └── visualization.py  # Vẽ ma trận attention so sánh giữa 2 mô hình
├── scripts/                  # Script chạy thực thi
│   ├── __init__.py
│   ├── preview_data.py       # Chạy thử loader và embedding trên dữ liệu thật
│   ├── train.py              # Huấn luyện mô hình
│   └── inference.py          # Dịch câu và xuất ma trận attention
├── tests/                    # Unit tests cho các module
│   └── test_dataset.py        # Kiểm thử reader, dataset và batch loader
├── INTERFACES.md              # Hợp đồng tích hợp tokenizer, model và metrics
├── requirements.txt          # Các thư viện phụ thuộc (torch, nltk, matplotlib, pyyaml, transformers, etc.)
├── README.md                 # Tài liệu hướng dẫn sử dụng repo
└── .gitignore
```

## Cài đặt thư viện

Trên máy không có GPU NVIDIA, chạy script sau để cài bản PyTorch CPU và các thư viện còn lại:

```bash
chmod +x install_cpu.sh
./install_cpu.sh
```

Nếu máy có GPU, chỉ cần chạy

```bash
pip install -r requirements.txt
```

## Dataset loader

Loader trả về `(src, tgt)` để dùng chung cho LSTM và Transformer:

- `src`: tensor `torch.long` trên CPU, shape `[batch_size, source_length]`.
- `tgt`: tensor `torch.long` trên CPU, shape `[batch_size, target_length]`, gồm
  `<s>`, nội dung câu đích và `</s>`.
- Hai phía được padding bên phải theo câu dài nhất trong batch, dùng PAD ID
  riêng của từng tokenizer. PAD ID không nhất thiết bằng `0`.
- Loader không dịch chuyển câu đích, tạo attention mask hoặc chuyển dữ liệu lên GPU.
  Trainer hiện tại tách `tgt[:, :-1]` và `tgt[:, 1:]`; mô hình/trainer phụ trách mask.

### Phân chia trách nhiệm

| Module | Thành phần | Trách nhiệm |
| --- | --- | --- |
| `src/data/readers.py` | `ParallelTextReader` | Đọc và kiểm tra hai file song ngữ, bỏ cặp trống |
| `src/data/tokenizer_protocol.py` | `TokenizerProtocol` | Mô tả giao diện tokenizer cần cung cấp |
| `src/data/validation.py` | `TokenizerValidator` | Kiểm tra metadata tokenizer và ID trước embedding |
| `src/data/dataset.py` | `TranslationDataset` | Encode một mẫu và thêm BOS/EOS cho câu đích |
| `src/data/batch.py` | `TranslationSample`, `TranslationBatch` | Cấu trúc mẫu và schema batch mở rộng |
| `src/data/collator.py` | `TranslationCollator` | Gom mẫu và padding batch |
| `src/data/loader.py` | `create_dataloader` | Kết nối các thành phần bằng PyTorch DataLoader |

### Giao diện tokenizer

Truyền riêng tokenizer tiếng Anh và tokenizer tiếng Việt. Mỗi tokenizer cung cấp:

```python
tokenizer.encode(text: str) -> list[int]
tokenizer.pad_id: int
tokenizer.bos_id: int  # <s>
tokenizer.eos_id: int  # </s>
tokenizer.unk_id: int
tokenizer.vocab_size: int  # Gồm toàn bộ special token
```

`encode()` chỉ trả ID nội dung: không thêm BOS/EOS hoặc padding; tự ánh xạ từ
không có trong vocabulary sang `<unk>`. Tokenizer không cần kế thừa class của
loader, chỉ cần cung cấp đúng giao diện. Nếu tokenizer của nhóm dùng tên hàm
khác, viết adapter cho giao diện này.

Model phải dùng `tokenizer.vocab_size` để khởi tạo embedding; không dùng số
dòng của file vocabulary trước khi append special token. Nếu tokenizer có
`__len__`, `len(tokenizer)` phải bằng `vocab_size`. Loader kiểm tra ID nằm
trong phạm vi này và không cho phép `encode()` chèn PAD/BOS/EOS.

Vocabulary chỉ được xây từ train (ngưỡng tần suất nhóm yêu cầu: `min_freq=2`),
sau đó tái sử dụng nguyên trạng cho validation/test. Tokenizer chịu trách nhiệm
lưu/tải mapping và append special token, bao gồm `<mask>`, sau vocabulary thông
thường theo quy ước của nhóm. Loader không xây vocabulary, không chèn `<mask>`
vào câu và không thay đổi ID đã lưu.

### Sử dụng với tokenizer của nhóm

Sau khi đã tạo hoặc tải `source_tokenizer` và `target_tokenizer`:

```python
from src.data import create_dataloader

train_loader = create_dataloader(
    source_path="data/raw/train-en-vi/train.en",
    target_path="data/raw/train-en-vi/train.vi",
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    batch_size=64,
    shuffle=True,
    seed=42,
)

validation_loader = create_dataloader(
    source_path="data/raw/dev-2012-en-vi/tst2012.en",
    target_path="data/raw/dev-2012-en-vi/tst2012.vi",
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    batch_size=64,
    shuffle=False,
)

for src, tgt in train_loader:
    # Chuyển thiết bị tại trainer, không thực hiện trong Dataset/collator.
    src, tgt = src.to(device), tgt.to(device)
    decoder_input = tgt[:, :-1]
    labels = tgt[:, 1:]
    # criterion = CrossEntropyLoss(ignore_index=target_tokenizer.pad_id)
```

Với test, dùng `data/raw/test-2013-en-vi/tst2013.en` và `tst2013.vi`, giữ
`shuffle=False`. Mặc định giữ batch cuối dù ít mẫu hơn `batch_size`.

Mỗi file chứa một câu trên một dòng, mã hóa UTF-8 (có thể có BOM). Hai file phải
có cùng số dòng; nếu lệch, reader báo lỗi kèm dòng đầu tiên không khớp. Nếu một
phía trống, bỏ cả cặp để giữ alignment. Không tự giải mã HTML entity, lowercase,
lọc câu dài, loại trùng hoặc chia lại tập: các bước chuẩn hóa đó thuộc pipeline
tiền xử lý chung của nhóm.

Để xem thống kê số cặp bị loại:

```python
from src.data import ParallelTextReader

reader = ParallelTextReader("data/raw/train-en-vi/train.en", "data/raw/train-en-vi/train.vi")
pairs = reader.read()
print(reader.stats)  # total_pairs, skipped_empty_pairs; có thêm thuộc tính kept_pairs
```

`num_workers=0` là mặc định để dễ chạy trên Windows. Nếu tăng giá trị này, đặt
code khởi chạy trong `if __name__ == "__main__":` và bảo đảm tokenizer có thể
pickle. Cùng seed và cùng cấu hình tạo lại loader sẽ cho cùng thứ tự shuffle;
khi duyệt lại cùng loader, thứ tự train thay đổi qua mỗi epoch.

### Batch mở rộng (tùy chọn)

Truyền `return_metadata=True` vào `create_dataloader()` để nhận dict gồm
`src`, `tgt`, `src_len`, `tgt_len`. Lengths là CPU `LongTensor[B]`, lấy trước
padding; `tgt_len` bao gồm BOS/EOS. Batch không sort theo độ dài, vì vậy dùng
`enforce_sorted=False` khi gọi `pack_padded_sequence`.

Truyền `return_tokens=True` để tự bật batch mở rộng và thêm `src_tokens`,
`tgt_tokens`. Chế độ này yêu cầu tokenizer cung cấp `tokenize(text) -> list[str]`
nhất quán với `encode()`. Danh sách token chữ giữ từ OOV và không chứa
BOS/EOS/PAD; token và ID vẫn đi cùng nhau khi shuffle.

Chế độ mặc định không đổi: `(src, tgt)`. Xem [INTERFACES.md](INTERFACES.md)
để biết schema, ví dụ embedding/LSTM và quy tắc tích hợp training/attention.

### Chạy thử và kiểm tra

Chạy từ thư mục gốc repo, trong môi trường đã cài PyTorch:

```bash
python -m scripts.preview_data --batch-size 4
python -m scripts.preview_data --batch-size 4 --return-tokens
python -m unittest discover -s tests -v
```

Script preview dùng adapter đơn giản với hai file vocabulary có sẵn để kiểm tra
luồng đọc/encode/padding trên dữ liệu thật. Đây không phải tokenizer huấn luyện
cuối cùng và không chứng minh vocabulary cũ đáp ứng `min_freq=2`. Adapter giữ
mapping cũ và chỉ append các special token còn thiếu trong bộ nhớ, không ghi
đè file vocabulary.
