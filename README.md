# neural-machine-translation-project
Implementation of seq2seq attention (LSTM) and Transformers from scratch to translate from English to Vietnamese.


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
│   │   ├── tokenizer.py      # Quản lý từ vựng, padding, mapping ID <-> token
│   │   └── dataset.py        # PyTorch Dataset và DataLoader
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
│   ├── train.py              # Huấn luyện mô hình
│   └── inference.py          # Dịch câu và xuất ma trận attention
├── tests/                    # Unit tests cho các module
├── requirements.txt          # Các thư viện phụ thuộc (torch, nltk, matplotlib, pyyaml, transformers, etc.)
├── README.md                 # Tài liệu hướng dẫn sử dụng repo
└── .gitignore
