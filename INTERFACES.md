# Giao diện dataset loader

## Tokenizer chính thức (giai đoạn 1)

```python
from src.data import TokenizerStorage, build_tokenizers

# Hai đường dẫn TRAIN đã chuẩn hóa, Anh trước, Việt sau:
source_tokenizer, target_tokenizer = build_tokenizers(en_path, vi_path, min_freq=2)
TokenizerStorage.save(source_tokenizer, "artifacts/tokenizers/source.json")
TokenizerStorage.save(target_tokenizer, "artifacts/tokenizers/target.json")

# Dev/test/inference chỉ tải lại mapping đã lưu:
source_tokenizer = TokenizerStorage.load("artifacts/tokenizers/source.json")
target_tokenizer = TokenizerStorage.load("artifacts/tokenizers/target.json")
```

Không fit dev/test. Tách bằng `split()`, giữ case và dấu câu: phần chuẩn hóa
phải nhất quán giữa train/dev/test và câu mới. Token dấu phẩy cần có khoảng
trắng bao quanh ở bước tiền xử lý của Vũ.

- `tokenize(text) -> list[str]`: token nội dung trước khi xử lý OOV.
- `encode(text) -> list[int]`: nội dung, không thêm BOS/EOS/PAD.
- `decode_tokens(ids) -> list[str]`: bỏ BOS/PAD/MASK, dừng tại EOS, giữ UNK.
- `decode(ids) -> str`: nối kết quả `decode_tokens()` bằng một dấu cách.
- Decode nhận một dãy ID hoặc tensor 1D, kể cả tensor GPU; không nhận tensor
  batch 2D. ID sai trước EOS báo lỗi. Từ OOV không khôi phục được cách viết.
- `vocabulary.token_to_id(token)`, `vocabulary.id_to_token(id)` cung cấp
  mapping trực tiếp. `vocabulary.tokens` là tuple theo thứ tự ID.
- Special token nằm cuối theo thứ tự PAD/BOS/EOS/UNK/MASK; nội dung train và
  đầu vào encode không được chứa nguyên văn các token dành riêng này.
- JSON lưu thứ tự ID và cấu hình; checkpoint phải dùng đúng hai JSON lúc
  train. Nếu xây lại vocab từ dữ liệu khác, ID có thể thay đổi.

Biên bản có dòng nói decode thành tiếng Anh; bài toán Anh→Việt cần encode
nguồn bằng `source_tokenizer` và decode dự đoán bằng `target_tokenizer`.
Phần model giữ API ID/tensor; wrapper nhận câu chữ mới gọi tokenize/encode.

## Tokenizer và kích thước vocabulary

Xuất kết quả encode/decode ra file (khác với lưu mapping):

```python
from src.data import TokenizerFileExporter

exporter = TokenizerFileExporter(source_tokenizer)
exporter.encode_file("sentences.en", "artifacts/outputs/encoded.jsonl")
exporter.decode_file("artifacts/outputs/encoded.jsonl", "artifacts/outputs/decoded.jsonl")
```

`encode_file` nhận UTF-8 text một câu mỗi dòng, ghi JSONL `{text, ids}`;
`decode_file` nhận JSONL có key `ids` và ghi `{ids, text}`. Cả hai trả số dòng
đã xuất; giữ dòng trống như sequence rỗng. Dùng đúng tokenizer của ngôn ngữ
ID; decode không dịch. `overwrite=True` cho phép thay output cũ sau khi xử lý
thành công; mặc định từ chối ghi đè. API trong bộ nhớ không thay đổi.

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

Thay các lời gọi giả định cũ khi ghép với code Trung:

```python
# Inference (nguồn là tiếng Anh):
src_tokens = source_tokenizer.tokenize(sentence)
src_indices = source_tokenizer.encode(sentence)

# Decode một câu dự đoán và một câu tham chiếu bằng tokenizer tiếng Việt:
pred_text = target_tokenizer.decode(pred_ids)
reference_text = target_tokenizer.decode(target_ids)
# Với attention, dùng target_tokenizer.decode_tokens(pred_ids).
```

`decode` trả chuỗi cho BLEU; `decode_tokens` trả danh sách cho visualization.
Không dùng `split_sentence`, `text_to_indices`, `indices_to_text` nếu không có
adapter riêng. Loader cũ không cần thay đổi; truyền `WordTokenizer` vào là dùng được.

Code đang dùng `for src, tgt in dataloader` tiếp tục chạy ở chế độ mặc định.
Chỉ bật chế độ mở rộng khi caller đã chuyển sang đọc các key của batch.
Không unpack dict bằng `for src, tgt in loader`.

Thay mọi giả định `pad_idx = 0` bằng PAD ID đúng của từng ngôn ngữ. Dùng
`target_tokenizer.pad_id` cho loss, `source_tokenizer.pad_id` cho mask nguồn.
Việc khai báo kích thước vocabulary thuộc model phải dùng mapping sau append.
