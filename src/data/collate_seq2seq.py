from src.data.datasets_seq2seq import *

def make_collate_fn(vocab: Vocab):
    def collate_fn(batch):
        batch_size = len(batch)

        src_lens = [len(x["src_ids"]) for x in batch]
        tgt_lens = [len(x["tgt_input_ids"]) for x in batch]

        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)

        src_batch = torch.full((batch_size, max_src_len), vocab.PAD_ID, dtype=torch.long)
        tgt_input_batch = torch.full((batch_size, max_tgt_len), vocab.PAD_ID, dtype=torch.long)
        tgt_output_batch = torch.full((batch_size, max_tgt_len), vocab.PAD_ID, dtype=torch.long)

        a_list, b_list, sum_list = [], [], []
        src_texts, tgt_texts = [], []

        for i, item in enumerate(batch):
            src = item["src_ids"]
            tgt_in = item["tgt_input_ids"]
            tgt_out = item["tgt_output_ids"]

            src_batch[i, :len(src)] = src
            tgt_input_batch[i, :len(tgt_in)] = tgt_in
            tgt_output_batch[i, :len(tgt_out)] = tgt_out

            a_list.append(item["a"])
            b_list.append(item["b"])
            sum_list.append(item["sum_"])
            src_texts.append(item["src_text"])
            tgt_texts.append(item["tgt_text"])

        return {
            "src_ids": src_batch, #numbers
            "src_lens": torch.tensor(src_lens, dtype=torch.long),
            "tgt_input_ids": tgt_input_batch,
            "tgt_output_ids": tgt_output_batch,
            "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
            "a": a_list,
            "b": b_list,
            "sum_": sum_list,
            "src_text": src_texts,
            "tgt_text": tgt_texts,
        }

    return collate_fn