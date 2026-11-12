import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer :
    def __init__(
        self, 
        model, 
        optimizer,
        criterion,
        device,
        clip_grad_norm=1.0
    ) :
        self.model = model
        self.optimizer = optimizer

        self.criterion = criterion # Thường là nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.device = device
        self.clip_grad_norm = clip_grad_norm

    def train_one_epoch(
        self, 
        dataloader,
    ) :
        self.model.train()
        total_loss = 0.

        for src, tgt in tqdm(dataloader, desc="Đang huấn luyện") :
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            # Chú ý về teacher forcing (đặc biệt là trong LSTM)
            output = self.model(
                src,
                tgt_input
            )

            # output shape: [batch_size, tgt_len - 1, vocab_size]
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_out = tgt_out.reshape(-1)

            loss = self.criterion(output, tgt_out)
            loss.backward()


            # Chống bùng nổ gradient (Gradient Clipping)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="Đang đánh giá") :
                src = src.to(self.device)
                tgt = tgt.to(self.device)


                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                # Chú ý có thể thêm teacher_forcing_ratio trong tương lai nếu cần
                output = self.model(
                    src, 
                    tgt_input
                )

                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt_out = tgt_out.reshape(-1)

                loss = self.criterion(output, tgt_out)
                total_loss += loss.item()


        return total_loss / len(dataloader)