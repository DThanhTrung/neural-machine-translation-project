from src.evaluation import calculate_bleu

import torch

def evaluate_bleu_score(model, dataloader, tokenizer, device):
    """Hàm chạy riêng để tính BLEU trên toàn bộ tập dữ liệu"""
    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for src, tgt in dataloader :
            src = src.to(device)
            tgt = tgt.to(device)

            # sinh câu dự đoán
            pred_ids, attn_matrix = model.generate(src) # Giả định mô hình có hàm generate

            # decode từ ID sang text và split theo dấu cách
            for pred_ids, tgt_ids in zip(pred_ids, tgt):
                pred_text = tokenizer.indices_to_text(pred_ids)
                ref_text = tokenizer.indices_to_text(tgt_ids)

                all_preds.append(pred_text.split())
                all_refs.append(ref_text.split())


    bleu_score = calculate_bleu(all_preds, all_refs)
    return bleu_score
