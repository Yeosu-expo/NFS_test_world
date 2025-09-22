import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import DataCollatorForLanguageModeling

def _load_wikitext(tokenizer, collator, max_length=None, split_size=None):
    ds = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]

    # 1) 토크나이즈 (패딩/트렁케이션 하지 않음; special tokens은 수동으로 EOS만 삽입)
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return out

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"]) 

    # 2) 연속 청크로 묶기 (문서 사이에 EOS 삽입)
    def group_texts(examples):
        eos_id = tokenizer.eos_token_id
        # 모든 시퀀스를 하나로 이어 붙이되, 각 시퀀스 사이에 EOS를 삽입
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids + [eos_id])
        # block 단위로 자르기
        total_len = (len(all_ids) // max_length) * max_length
        all_ids = all_ids[:total_len]
        input_ids = [all_ids[i : i + max_length] for i in range(0, total_len, max_length)]
        labels = [ids.copy() for ids in input_ids]
        # 3) pad 마스킹 (안전장치: pad가 존재하면 -100)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        if pad_id != -1:
            for row in labels:
                for j in range(len(row)):
                    if row[j] == pad_id:
                        row[j] = -100
        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(group_texts, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "labels"]) 

    if split_size is not None:
        print(f"[Datasets] len (grouped blocks): {len(ds)}")
        ds = ds.select(range(min(split_size, len(ds))))
    return ds

def get_data_loader(batch_size:int, max_length:int, split_size:int, tokenizer, world_size:int, rank, is_multi=False) -> DataLoader:
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataset = _load_wikitext(tokenizer, collator, max_length=max_length, split_size=split_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, num_replicas=int(world_size), rank=int(rank)),
        num_workers=int(world_size)
    )

    return train_loader