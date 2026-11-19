# neural-machine-translation-project

Implementation of seq2seq attention (LSTM) and Transformers from scratch to translate from English to Vietnamese.

Cấu trúc tổng thể dự kiến của dự án được mô tả dưới đây. Nhánh
`huy-dataset-loader` đã triển khai tokenizer whitespace, xây/lưu/tải vocabulary,
pipeline loader trong `src/data/`, script chạy thử và kiểm thử. Chuẩn hóa dữ
liệu do phần tiền xử lý của nhóm cung cấp; cấu hình và các mô hình chưa có
đầy đủ trên nhánh này.

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
│   │   ├── tokenizer.py      # WordTokenizer: tokenize, encode, decode
│   │   ├── vocabulary.py     # Mapping token <-> ID và special token
│   │   ├── vocabulary_builder.py # Xây vocab từ hai file train, min_freq=2
│   │   ├── tokenizer_storage.py # Lưu/tải mapping bằng JSON
│   │   ├── tokenizer_export.py # Xuất kết quả encode/decode ra JSONL
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
│   ├── build_tokenizers.py   # CLI xây tokenizer từ hai đường dẫn train
│   ├── export_tokenizer.py   # CLI xuất ID/câu decode ra file
│   ├── preview_data.py       # Chạy thử loader và embedding trên dữ liệu thật
│   ├── train.py              # Huấn luyện mô hình
│   └── inference.py          # Dịch câu và xuất ma trận attention
├── tests/                    # Unit tests cho các module
│   ├── test_dataset.py        # Kiểm thử reader, dataset và batch loader
│   ├── test_tokenizer.py      # Kiểm thử tokenizer, JSON và tích hợp loader
│   └── test_tokenizer_export.py # Kiểm thử xuất file encode/decode
├── artifacts/tokenizers/      # JSON sinh cục bộ, được gitignore
├── artifacts/outputs/         # Kết quả encode/decode JSONL, được gitignore
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

## Tokenizer và vocabulary (giai đoạn 1)

`WordTokenizer` dùng `text.split()`: không lowercase, không tự tách dấu câu,
không dùng BPE. Với tiếng Việt, đây là mức âm tiết theo khoảng trắng. Phần
tiền xử lý của Vũ cần đưa dữ liệu về cùng chuẩn, ví dụ `hello , world` để dấu
phẩy là một token; câu mới khi inference cũng cần qua cùng chuẩn hóa.

| Module | Thành phần | Trách nhiệm |
| --- | --- | --- |
| `src/data/vocabulary.py` | `Vocabulary` | Mapping hai chiều cố định, ID liên tiếp và special token |
| `src/data/vocabulary_builder.py` | `VocabularyBuilder`, `build_tokenizers` | Đếm token từ train, lọc tần suất và xây hai vocab riêng |
| `src/data/tokenizer.py` | `WordTokenizer` | Tách token, encode ID, decode token/câu |
| `src/data/tokenizer_storage.py` | `TokenizerStorage` | Save/load mapping và cấu hình bằng JSON UTF-8 |
| `src/data/tokenizer_export.py` | `TokenizerFileExporter` | Xuất kết quả chuyển đổi từng câu ra JSONL |

`build_tokenizers(en_path, vi_path, min_freq=2)` kiểm tra hai file train đúng
alignment, bỏ cả cặp trống rồi đếm tần suất riêng cho mỗi ngôn ngữ. Chỉ giữ
token xuất hiện từ 2 lần. ID thông thường được sắp theo tần suất giảm dần,
hòa thì theo thứ tự Unicode; sau đó append `<pad>`, `<s>`, `</s>`, `<unk>`,
`<mask>` theo đúng thứ tự. Không đưa các chuỗi special token nguyên văn vào
corpus nội dung, vì chúng được dành riêng cho pipeline.

CLI nhận **hai đường dẫn train**. Ví dụ chạy bằng dữ liệu đang có trong repo:

```bash
python -m scripts.build_tokenizers --en-path data/raw/train-en-vi/train.en --vi-path data/raw/train-en-vi/train.vi --min-freq 2
```

Khi dữ liệu đã chuẩn hóa của Vũ sẵn sàng, thay hai đường dẫn trên bằng file
Anh/Việt thuộc **train** trong `data/processed/`; không lấy dev/test để fit.
Đường dẫn cụ thể theo cấu trúc Vũ bàn giao. Không ghi đè mapping dùng với
checkpoint cũ bằng vocabulary xây lại, vì các hàng embedding phụ thuộc ID.

Mặc định tạo `artifacts/tokenizers/source.json` và `target.json` (thay đổi bằng
`--output-dir`). Đây là file sinh được nên đã gitignore. Mỗi JSON chứa
`format_version`, `tokenization`, `min_freq`, `id_to_token`. Vị trí trong danh
sách chính là ID; load khôi phục đúng thứ tự mà không fit hoặc sắp xếp lại.
`min_freq=null` chỉ dành cho vocabulary được khởi tạo thủ công; CLI xây từ
train lưu ngưỡng thực tế. Model/checkpoint cần đi kèm đúng hai JSON đã dùng.

```python
from src.data import TokenizerStorage, build_tokenizers

# Fit một lần từ train:
source_tokenizer, target_tokenizer = build_tokenizers(en_path, vi_path, min_freq=2)
TokenizerStorage.save(source_tokenizer, "artifacts/tokenizers/source.json")
TokenizerStorage.save(target_tokenizer, "artifacts/tokenizers/target.json")

# Những lần chạy sau, bao gồm dev/test/inference, chỉ load:
source_tokenizer = TokenizerStorage.load("artifacts/tokenizers/source.json")
target_tokenizer = TokenizerStorage.load("artifacts/tokenizers/target.json")

source_ids = source_tokenizer.encode("I love learning .")
# Sau khi model sinh pred_ids tiếng Việt:
# predicted_text = target_tokenizer.decode(pred_ids)
# predicted_tokens = target_tokenizer.decode_tokens(pred_ids)
```

`encode(text)` trả `list[int]` nội dung, từ OOV thành `unk_id`; không chèn
BOS/EOS/PAD. `decode_tokens(ids)` trả `list[str]`, bỏ BOS/PAD/MASK, dừng tại
EOS và giữ `<unk>`. `decode(ids)` nối chúng thành chuỗi bằng một dấu cách.
Chấp nhận một dãy ID hoặc tensor 1D CPU/GPU, không nhận cả batch; duyệt từng
hàng khi decode batch. ID không hợp lệ trước EOS sẽ báo lỗi. Decode không
khôi phục được từ OOV ban đầu hoặc khoảng trắng nguyên bản.

### Xuất kết quả encode/decode ra file

Hai JSON trong `artifacts/tokenizers/` lưu **mapping**. Để lưu **kết quả chuyển
đổi từng câu**, dùng `TokenizerFileExporter` hoặc CLI sau (chạy sau khi build
tokenizer):

```bash
python -m scripts.export_tokenizer encode --tokenizer artifacts/tokenizers/source.json --input data/raw/dev-2012-en-vi/tst2012.en --output artifacts/outputs/dev.en.encoded.jsonl
python -m scripts.export_tokenizer decode --tokenizer artifacts/tokenizers/source.json --input artifacts/outputs/dev.en.encoded.jsonl --output artifacts/outputs/dev.en.decoded.jsonl
python -m scripts.export_tokenizer encode --tokenizer artifacts/tokenizers/target.json --input data/raw/dev-2012-en-vi/tst2012.vi --output artifacts/outputs/dev.vi.encoded.jsonl
python -m scripts.export_tokenizer decode --tokenizer artifacts/tokenizers/target.json --input artifacts/outputs/dev.vi.encoded.jsonl --output artifacts/outputs/dev.vi.decoded.jsonl
```

JSONL là UTF-8, mỗi dòng là một object JSON độc lập:

```json
{"text": "I learn .", "ids": [10, 20, 30]}
```

File decoded có schema `{"ids": [...], "text": "..."}`. ID trong ví dụ là
minh họa; kết quả thực tế theo mapping đã load. Decode cũng nhận file dự đoán
của model nếu mỗi dòng chứa `{"ids": [...]}`. Phải dùng đúng tokenizer đã
encode/sinh ID; decode không dịch ngôn ngữ.

```python
from src.data import TokenizerFileExporter, TokenizerStorage

tokenizer = TokenizerStorage.load("artifacts/tokenizers/source.json")
exporter = TokenizerFileExporter(tokenizer)
count = exporter.encode_file("sentences.txt", "artifacts/outputs/encoded.jsonl")
exporter.decode_file("artifacts/outputs/encoded.jsonl", "artifacts/outputs/decoded.jsonl")
```

File encode giữ nguyên số/thứ tự dòng, kể cả câu trống (ID `[]`), để không lệch
alignment. Khác với reader huấn luyện, exporter không tự loại cặp trống.
`encode()`/`decode()` gốc vẫn chỉ chuyển đổi trong bộ nhớ; chức năng ghi file
nằm ở module riêng để loader không ghi file mỗi lần lấy mẫu.

Đầu ra đã tồn tại cần `--overwrite` (API: `overwrite=True`); input và output
không được cùng đường dẫn. Nếu lỗi JSON/ID hoặc encode ở một dòng, thông báo
có số dòng và output cũ được giữ nguyên. Output mới chỉ được thay vào khi
toàn bộ thao tác thành công. Các file sinh trong `artifacts/outputs/` không
được tự đưa lên Git.

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

Vocabulary được xây riêng bằng `build_tokenizers()` từ train (`min_freq=2`),
sau đó tái sử dụng nguyên trạng cho validation/test. `TokenizerStorage` phụ
trách lưu/tải mapping. Loader không tự xây vocabulary, không chèn `<mask>`
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
python -m scripts.build_tokenizers --en-path data/raw/train-en-vi/train.en --vi-path data/raw/train-en-vi/train.vi
python -m scripts.preview_data --batch-size 4
python -m scripts.preview_data --batch-size 4 --return-tokens
python -m unittest discover -s tests -v
```

Script preview load hai JSON tokenizer thật, không dùng adapter vocabulary cũ
và không tự fit lại. Chọn vị trí JSON bằng `--tokenizer-dir`. Mặc định preview
train raw hiện có; để xem dữ liệu khác, truyền cả `--source-path` và
`--target-path`. Preview in thống kê, batch, encode/decode và thử embedding.
Việc chạy thử bằng train raw xác nhận pipeline kỹ thuật; vocab huấn luyện cuối
cùng cần xây từ train đã được Vũ chuẩn hóa.
