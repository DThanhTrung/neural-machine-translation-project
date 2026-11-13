import torch
import torch.nn as nn
from tqdm import tqdm

import os
from safetensors.tensorflow import save_file, load_file

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


    def save_checkpoint(
        self,
        checkpoint_filename: str,
        cur_epoch: int,
        train_loss: float,
        val_loss: float,
        train_bleu: float,
        val_bleu: float
    ) :
        """
        Lưu lại checkpoint khi huấn luyện, 
        tách biệt thành safetensors, optimizer state và metadata đi kèm
        """

        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'cur_epoch': cur_epoch,

            'train_loss': train_loss,
            'val_loss': val_loss,

            'train_bleu': train_bleu,
            'val_bleu': val_bleu
        }, checkpoint_filename)


    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_filename: str
    ) :
        # TODO
        raise NotImplementedError("You must implement load_checkpoint method")


    def save_model_safetensors(
        self, 
        save_path: str
    ) :
        """
        Save model weights and architecture configuration as safetensors.
        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Lấy state_dict của mô hình
        state_dict = self.model.state_dict()

        # Đảm bảo tất cả tensor đều nằm trên CPU và ở dạng contiguous (bắt buộc đối với safetensors)
        cpu_state_dict = {
            k: v.cpu().contiguous() 
            for k, v in state_dict.items()
        }

        # Lưu file
        save_file(
            tensors=cpu_state_dict, 
            filename=save_path
        )

        print(f"Đã lưu mô hình an toàn bằng safetensors tại: {save_path}")


    @classmethod
    def load_model_safetensors(
        cls,
        model, 
        load_path, 
        device
    ) :
        # Load state_dict từ file safetensors
        state_dict = load_file(load_path)

        # Đưa vào mô hình
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"Đã nạp trọng số thành công từ: {load_path}")
        return model