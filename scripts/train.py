
from src.training import Trainer
from .utils import evaluate_bleu_score

import torch
import torch.nn as nn
import torch.optim as optim

import os
import tqdm

NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
EMBED_DIM = 256
HIDDEN_DIM = 512


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

    print("Đã tải dữ liệu huấn luyện, kiểm định và tokenizer xong")
    print("=" * 50)

    # các thông số (giả sử)
    src_vocab_size = 30000 # Ví dụ kích thước từ vựng giả định
    tgt_vocab_size = 30000
    pad_idx = 0            # Index của thẻ <pad> trong từ vựng

    # Khởi tạo model
    # TODO
    model = None

    # Cấu hình Optimizer và Loss Function
    # Bắt buộc truyền ignore_index=pad_idx để mô hình không học dự đoán các token padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        clip_grad_norm=1.0
    )

    print("Bắt đầu huấn luyện")
    train_losses = []
    val_losses = []

    train_bleus = []
    val_bleus = []

    for epoch in tqdm.tnrange(NUM_EPOCHS, desc='Đang huấn luyện') :
        model.train()

        train_loss = trainer.train_one_epoch(
            train_loader
        )

        val_loss = trainer.evaluate(
            validation_loader
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if 0 == epoch % 5 :
            train_bleu = evaluate_bleu_score(
                model=model,
                dataloader=train_loader,
                tokenizer=tokenizer,
                device=device
            )

            val_bleu = evaluate_bleu_score(
                model=model,
                dataloader=validation_loader,
                tokenizer=tokenizer,
                device=device
            )

            train_bleus.append(train_bleu)
            val_bleus.append(val_bleu)

            checkpoint_filename = os.path.join('checkpoints', f'model_epoch_{str(epoch).zfill(3)}.pt')
            trainer.save_checkpoint(
                checkpoint_filename=checkpoint_filename,
                cur_epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_bleu=train_bleu,
                val_bleu=val_bleu
            )            


    model_name : str = 'model.safetensors'
    trainer.save_model_safetensors(
        os.path.join('model_weights', model_name)
    )


if __name__ == "__main__" :
    main()