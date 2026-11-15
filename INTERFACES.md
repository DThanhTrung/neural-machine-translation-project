# Giao diện dataset loader

## Tokenizer và kích thước vocabulary

Mỗi ngôn ngữ có một tokenizer riêng, đã fit trên train và giữ nguyên trong
suốt quá trình tạo/duyệt loader. Tokenizer cung cấp:

```python
encode(text: str) -> list[int]
vocab_size: int
pad_id: int
bos_id: int
eos_id: int
unk_id: int
```

- `vocab_size` là số ID của **toàn bộ vocabulary sau khi thêm special token**.
  Mapping cần dùng ID liên tiếp từ `0` đến `vocab_size - 1`.
- Nếu tokenizer cung cấp `__len__`, `len(tokenizer)` phải bằng `vocab_size`.
  Không lấy số dòng của file vocabulary cũ làm kích thước embedding.
- `encode()` trả ID nội dung, có thể có UNK, không chứa PAD/BOS/EOS.
- PAD/BOS/EOS/UNK phải có ID riêng biệt trong `[0, vocab_size)`.
- Vocabulary bao gồm `<mask>` nếu tokenizer của nhóm dành sẵn token này.
  Loader không sử dụng hoặc chèn `<mask>`.
- Loader kiểm tra metadata khi tạo Dataset và từng ID nội dung khi lấy mẫu;
  tokenizer thiếu `vocab_size` hoặc có ID sai sẽ nhận lỗi có ngữ cảnh.
- Loader không fit, lưu/tải vocabulary, không sửa hoặc thêm token vào tokenizer.

Khởi tạo embedding và loss:

```python
source_embedding = torch.nn.Embedding(
    source_tokenizer.vocab_size, embedding_dim,
    padding_idx=source_tokenizer.pad_id,
)
target_embedding = torch.nn.Embedding(
    target_tokenizer.vocab_size, embedding_dim,
    padding_idx=target_tokenizer.pad_id,
)
criterion = torch.nn.CrossEntropyLoss(ignore_index=target_tokenizer.pad_id)
```

## Chế độ mặc định: tương thích Trainer hiện tại

```python
loader = create_dataloader(source_path, target_path, source_tokenizer, target_tokenizer)
for src, tgt in loader:
    decoder_input = tgt[:, :-1]
    labels = tgt[:, 1:]
```

`src` và `tgt` là tensor CPU `torch.long`, batch-first, padding bên phải.
Nguồn chỉ chứa ID nội dung; đích chứa `[BOS, nội dung, EOS]` trước padding.

## Batch mở rộng: độ dài cho LSTM

```python
loader = create_dataloader(
    source_path, target_path, source_tokenizer, target_tokenizer,
    return_metadata=True,
)
for batch in loader:
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        source_embedding(batch["src"].to(device)),
        batch["src_len"],
        batch_first=True,
        enforce_sorted=False,
    )
```

| Key | Kiểu/shape | Ý nghĩa |
| --- | --- | --- |
| `src` | `LongTensor[B, S]` | ID nguồn đã padding |
| `tgt` | `LongTensor[B, T]` | ID đích, BOS/EOS và padding |
| `src_len` | CPU `LongTensor[B]` | Số ID nguồn trước padding |
| `tgt_len` | CPU `LongTensor[B]` | Số ID đích trước padding, **gồm BOS/EOS** |

Độ dài lấy trực tiếp từ các mẫu trước padding; không phụ thuộc giá trị PAD ID.
Batch không được sort theo độ dài; dùng `enforce_sorted=False` khi pack.
Số nhãn hợp lệ sau dịch chuyển là `tgt_len - 1` (gồm EOS, không gồm BOS).
Trainer phụ trách chuyển tensor cần thiết sang device; giữ lengths trên CPU khi pack.

## Token chữ cho đánh giá/trực quan hóa

```python
loader = create_dataloader(
    source_path, target_path, source_tokenizer, target_tokenizer,
    return_tokens=True,
)
for batch in loader:
    source_tokens = batch["src_tokens"]
    reference_tokens = batch["tgt_tokens"]
```

`return_tokens=True` tự bật batch mở rộng, thêm hai key
`src_tokens`, `tgt_tokens` kiểu `list[list[str]]`.

Hai tokenizer phải cung cấp thêm `tokenize(text: str) -> list[str]`, trả đúng
chuỗi token nội dung mà `encode()` sử dụng, trước khi ánh xạ từ lạ thành UNK.
Loader kiểm tra số token bằng số ID nội dung. Tokenizer phải cho kết quả xác
định để hai lời gọi nhất quán. Không tự `.split()` trong loader.

- Token chữ giữ cách viết của từ OOV, không bị đổi thành chuỗi `<unk>`.
- Không thêm BOS/EOS/PAD vào danh sách token chữ.
- `src_tokens[i]` tương ứng `src[i, :src_len[i]]`.
- `tgt_tokens[i]` tương ứng `tgt[i, 1:tgt_len[i]-1]`.
- Token và tensor đi cùng một sample qua shuffle; không cần đọc lại raw file.
- `tgt_tokens` là câu tham chiếu, không phải bản dịch do model sinh.
  Khi vẽ attention cho bản dịch sinh ra, dùng token dự đoán thực tế và loại
  các hàng/cột BOS/EOS/PAD tương ứng với quy ước của mô hình.

## Tích hợp với nhánh training/metrics

Code đang dùng `for src, tgt in dataloader` tiếp tục chạy ở chế độ mặc định.
Chỉ bật chế độ mở rộng khi caller đã chuyển sang đọc các key của batch.
Không unpack dict bằng `for src, tgt in loader`.

Thay mọi giả định `pad_idx = 0` bằng PAD ID đúng của từng ngôn ngữ. Dùng
`target_tokenizer.pad_id` cho loss, `source_tokenizer.pad_id` cho mask nguồn.
Việc khai báo kích thước vocabulary thuộc model phải dùng mapping sau append.
