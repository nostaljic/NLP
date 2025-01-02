from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter

src_vocab_size = 8000  # 소스 단어 집합 크기
tgt_vocab_size = 8000  # 타겟 단어 집합 크기
src_max_length = 512  # 소스 최대 문장 길이
tgt_max_length = 512  # 타겟 최대 문장 길이
embed_size = 512  # 임베딩 크기
num_heads = 8  # 헤드 수
num_layers = 6  # Encoder, Decoder 레이어 수
hidden_dim = 2048  # FFN의 히든 크기
dropout = 0.1  # 드롭아웃 비율
num_epochs = 1

# 입력 데이터
src = torch.randint(0, src_vocab_size, (2, 10))  # (batch_size, src_seq_len)
tgt = torch.randint(0, tgt_vocab_size, (2, 10))  # (batch_size, tgt_seq_len)

# Transformer 모델 생성
model = Transformer(src_vocab_size, tgt_vocab_size, src_max_length, tgt_max_length,
                    embed_size, num_heads, num_layers, hidden_dim, dropout)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)
model = model.to(device)

# Transformer 실행
out = model(src, tgt)
print("Transformer 출력:", out.shape)  # 출력 크기: (batch_size, tgt_seq_len, tgt_vocab_size)

raw_datasets = load_dataset("Heerak/ko_en_parallel_dataset")
print("Dataset splits:", raw_datasets) 
raw_datasets['train'] = raw_datasets['train'].select(range(1000000))
train_data = raw_datasets['train'].select(range(1000000))

##########################
# 3. 간단 토크나이저 및 어휘사전 구성
##########################
# 여기서는 '띄어쓰기 단위'로 split하는 간단 토크나이저를 사용합니다.
# 실제로는 SentencePiece/BPE 등을 사용합시다!

# (1) 파라미터 설정
max_length = 500       # 문장 최대 길이
src_lang = "ko"
tgt_lang = "en"

special_tokens = ["<kpep_end>", "<kpep_start>", "<kpep_eos>", "<kpep_unk>"]
pad_idx = 0
bos_idx = 1
eos_idx = 2
unk_idx = 3

def tokenize_ko(text):
    return text.strip().split()

def tokenize_en(text):
    return text.strip().split()

##########################
# 3-1. 전체 단어 빈도 계산
##########################
from collections import Counter
src_counter = Counter()
tgt_counter = Counter()

for row in train_data:
    ko_text = row[src_lang]
    en_text = row[tgt_lang]
    src_counter.update(tokenize_ko(ko_text))
    tgt_counter.update(tokenize_en(en_text))

num_epochs = 2   # 간단히 2 epoch만

for row in train_data:
    ko_text = row[src_lang]
    en_text = row[tgt_lang]
    src_counter.update(tokenize_ko(ko_text))
    tgt_counter.update(tokenize_en(en_text))

# 많이 나온 단어 순으로 vocab_size-4개를 추출 (특수 토큰 4개)
src_vocab = special_tokens + [w for w, _ in src_counter.most_common(src_vocab_size - 4)]
tgt_vocab = special_tokens + [w for w, _ in tgt_counter.most_common(tgt_vocab_size - 4)]

src_word2idx = {w: i for i, w in enumerate(src_vocab)}
tgt_word2idx = {w: i for i, w in enumerate(tgt_vocab)}

##########################
# 3-2. 문장 → 정수 인덱스 변환 함수
##########################
def encode_src(text):
    tokens = tokenize_ko(text)[:max_length - 2]  # <bos>, <eos> 고려
    encoded = [bos_idx] + [
        src_word2idx.get(w, unk_idx) for w in tokens
    ] + [eos_idx]
    return encoded

def encode_tgt(text):
    tokens = tokenize_en(text)[:max_length - 2]
    encoded = [bos_idx] + [
        tgt_word2idx.get(w, unk_idx) for w in tokens
    ] + [eos_idx]
    return encoded

def pad_sequence(seq, max_len):
    seq = seq[:max_len]
    seq += [pad_idx] * (max_len - len(seq))
    return seq

##########################
# 3-3. 전처리 함수 정의
##########################
def preprocess_function(examples):
    # examples는 여러 샘플을 batch 단위로 포함
    ko_texts = examples[src_lang]
    en_texts = examples[tgt_lang]

    src_encoded = [encode_src(ko) for ko in ko_texts]
    tgt_encoded = [encode_tgt(en) for en in en_texts]

    src_padded = [pad_sequence(seq, max_length) for seq in src_encoded]
    tgt_padded = [pad_sequence(seq, max_length) for seq in tgt_encoded]

    return {
        "src": src_padded,
        "tgt": tgt_padded
    }

##########################
# 3-4. 데이터셋에 mapping 적용
##########################
column_names = train_data.column_names  # ['ko', 'en'] 등
train_dataset = raw_datasets["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=column_names
)
valid_dataset = None
if "validation" in raw_datasets:
    valid_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names
    )

print("train_dataset[0]:", train_dataset[0])

##########################
# 3-5. DataLoader
##########################
def collate_fn(batch):
    # 이미 pad 처리가 되어 있으므로, 텐서 변환만
    src = torch.tensor([ex['src'] for ex in batch], dtype=torch.long)
    tgt = torch.tensor([ex['tgt'] for ex in batch], dtype=torch.long)
    return src, tgt

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = None
if valid_dataset:
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9
)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        src_batch = src_batch.to(device)  # (batch_size, seq_len)
        tgt_batch = tgt_batch.to(device)  # (batch_size, seq_len)
        
        # 디코더 입력/출력 분리
        # e.g. [bos, w1, w2, ..., eos] -> input
        #      [w1, w2, ..., eos, pad] -> label (한 토큰 뒤로 shift)
        tgt_in = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]
        
        logits = model(src_batch, tgt_in)  # (batch_size, seq_len-1, vocab_size)
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # (batch_size*(seq_len-1), vocab_size)
            tgt_out.reshape(-1)                  # (batch_size*(seq_len-1))
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Step {batch_idx+1} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"==> Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")

##########################
# 6. 간단 번역 테스트
##########################
model.eval()
def make_pad_mask(x, pad_idx=0):
    return (x != pad_idx).unsqueeze(1).unsqueeze(2)  # True: 사용, False: 마스킹할 위치
    
def make_no_peak_mask(x):
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        return causal_mask.unsqueeze(0).unsqueeze(1)
    
def generate(model, src_sentence, max_len=40):
    # 1) 소스 인코딩
    src_encoded = encode_src(src_sentence)
    src_padded = pad_sequence(src_encoded, max_length)
    src_tensor = torch.tensor([src_padded], dtype=torch.long, device=device)
    
    enc_out = model.encoder(src_tensor, make_pad_mask(src_tensor))
    
    # 2) 디코더의 입력으로 <kpep_start> 만 있는 상태에서 시작
    generated = torch.tensor([[bos_idx]], device=device)
    
    for _ in range(max_len):
        tgt_mask = make_pad_mask(generated) & make_no_peak_mask(generated)
        dec_out = model.decoder(generated, enc_out,
                                make_pad_mask(src_tensor),
                                tgt_mask)
        
        # 마지막 타임스텝의 argmax
        next_token = dec_out[:, -1, :].argmax(dim=-1)
        
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == eos_idx:
            break
    
    return generated.squeeze(0).tolist()


test_ko = "우리는 국가의 혁신을 즐길 수도 있습니다."
with torch.no_grad():
    gen_ids = generate(model, test_ko)

decoded_tokens = [tgt_vocab[t] for t in gen_ids]
print("Generated token IDs:", gen_ids)
print("\n=====================================")
print("[Korean] ",test_ko)
print("[English] ", " ".join(decoded_tokens))
print("=====================================\n")