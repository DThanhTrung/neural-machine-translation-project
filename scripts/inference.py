from src.training import Trainer
from src.evaluation import visualize_attention_matrix

import torch
import matplotlib.pyplot as plt


def translate_sentence(model, tokenizer, sentence, device, max_len=50):
    """
    Hàm thực hiện dịch câu và trả về danh sách từ đích kèm ma trận attention
    """
    model.eval()

    src_tokens = tokenizer.split_sentence(sentence)
    src_indices = tokenizer.text_to_indices(src_tokens)
    src_tensors = torch.tensor(
        data=src_indices,
        dtype=torch.int64
    ).unsqueeze(0).to(device=device)  # shape = (1, sentence_len)

    with torch.no_grad() :
        pred_ids, attn_matrix = model.generate(src_tensors)

    pred_tokens = tokenizer.indices_to_text(pred_ids)

    return src_tokens, pred_tokens, attn_matrix


def main() :
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    print("=" * 50)

    # Chuẩn bị dữ liệu và tokenizer
    print("Chuẩn bị tải dữ liệu và tokenizer")
    tokenizer = None

    # ... thêm vào trong tương lai
    # TODO
    train_loader, validation_loader, test_loader = None, None, None
    print("Đã tải dữ liệu kiểm tra và tokenizer xong")
    print("=" * 50)

    # Nạp mô hình từ tệp .safetensors
    model = None            # Khởi tạo mô hình trước
    model = Trainer.load_model_safetensors(
        model=model,
        load_path='',
        device=device
    )

    src_tokens, pred_tokens, attn_matrix = translate_sentence(
        model=model, 
        tokenizer=tokenizer, 
        sentence="This is an English sentence", 
        device=device
    )


    figure, axes = visualize_attention_matrix(
        src_sentence=src_tokens,
        pred_sentence=pred_tokens,
        attn_matrix=attn_matrix
    )
    plt.show()


if __name__ == "__main__" :
    main()