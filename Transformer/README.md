# Transformer Implementation in PyTorch

이 저장소는 2017년 구글 브레인 팀이 발표한 논문 [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762)에서 소개된 **Transformer** 아키텍처를 **PyTorch**로 구현한 예시 코드입니다.

- **파일 구조**  
  - `MultiHeadAttention`, `FeedForward`, `LayerNormalization` 등은 “Attention Is All You Need” 논문에서 소개된 주요 모듈을 각각 클래스로 구현.
  - `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, `Transformer` 클래스를 통해 최종 완전한 Transformer 모델 구조를 구성.

---

## 논문에서 소개된 주요 수식

### 1. Scaled Dot-Product Attention

Transformer의 핵심인 어텐션(Attention) 메커니즘은 쿼리 \(Q\)와 키 \(K\), 값 \(V\) 사이의 관계를 학습합니다.  
아래 식은 어텐션의 가장 기초적인 형태인 **Scaled Dot-Product Attention**을 나타냅니다:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

- \( d_k \)는 키 벡터의 차원.
- 논문에서는 \( d_k = \frac{d_{\text{model}}}{h} \) (멀티 헤드 수 \( h \)로 나눈 값).

### 2. Multi-Head Attention

하나의 **Scaled Dot-Product Attention**만 사용하는 대신, 여러 개의 헤드(Heads)로 병렬 계산하여 정보 표현력을 늘립니다.  
논문에서의 멀티헤드 어텐션( \( \text{MultiHead} \) ) 개념은 아래와 같은 식으로 표현됩니다:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]
\[
\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

- \( QW_i^Q \), \( KW_i^K \), \( VW_i^V \)는 쿼리/키/값에 대한 선형 변환.
- \( W^O \)는 각 헤드의 출력을 합쳐주는 선형 변환.

### 3. Position-wise Feed-Forward Network (FFN)

인코더와 디코더의 각 레이어 내부에는 “Position-wise Feed-Forward Network”가 들어있습니다.  
이는 시퀀스의 각 위치마다 독립적으로 동일한 FFN을 적용하는 구조로, 간단히 아래와 같이 표현할 수 있습니다.

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

### 4. Residual Connection & Layer Normalization

Transformer의 각 하위 모듈(어텐션, FFN) 계산 결과에는 **Residual Connection**을 적용하고, 그 뒤 **Layer Normalization**을 수행합니다.

\[
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon}\cdot \gamma + \beta
\]

---

## 주요 클래스별 설명

### 1. `MultiHeadAttention(nn.Module)`
- **역할**:  
  쿼리( \(Q\) ), 키( \(K\) ), 값( \(V\) ) 텐서에 대해서 Scaled Dot-Product Attention을 여러 헤드로 수행한 뒤, 다시 합치는 모듈입니다.
- **구성**:
  - \(\text{values}, \text{keys}, \text{queries}\): 쿼리/키/값에 대한 선형 변환 레이어 ( \(W_Q, W_K, W_V\) ).
  - `scaled_dot_product_attention`: \(\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V\) 연산 수행.
  - 최종적으로 모든 헤드 출력(`Concat`)을 하나로 합친 뒤 `fc_out`( \(W^O\) )으로 변환.

```python
class MultiHeadAttention(nn.Module):
    ...
    def scaled_dot_product_attention(...):
        # scores = Q @ K^T / sqrt(d_k)
        # attention = softmax(scores)
        # out = attention @ V
        return out, attention
```

### 2. `FeedForward(nn.Module)`
- **역할**:  
  위치별로 독립적으로 동작하는 2층 MLP (FFN)을 구현한 모듈입니다.
  \[
    \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
  \]
- **구성**:  
  - `fc1, fc2`: 두 개의 Linear 레이어와 ReLU 활성화 함수.
  - `dropout`: 학습 시 과적합을 방지하기 위한 드롭아웃.

```python
class FeedForward(nn.Module):
    ...
    def forward(self, x):
        # x -> fc1 -> ReLU -> dropout -> fc2
        return self.fc2(...)
```

### 3. `LayerNormalization(nn.Module)`
- **역할**:  
  Residual Connection 이후, 레이어 마지막에 적용되는 Layer Normalization을 수행합니다.
- **구성**:  
  - \(\gamma, \beta\): 정규화된 텐서에 곱/더해줄 학습 파라미터.
  - \( \mu, \sigma \): 입력 텐서 x의 평균/표준편차를 채널 마지막 차원 기준으로 계산.

```python
class LayerNormalization(nn.Module):
    ...
    def forward(self, x):
        # mean = x.mean(dim=-1, keepdim=True)
        # std = x.std(dim=-1, keepdim=True)
        # return gamma * (x - mean) / (std + eps) + beta
```

### 4. `EncoderLayer(nn.Module)`
- **역할**:  
  하나의 인코더 레이어를 나타내며, 다음 순서를 가집니다:
  1. Self-Attention → Residual & LayerNorm
  2. Feed Forward → Residual & LayerNorm

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        # 1) Self-Attention (x, x, x) -> Residual & Norm
        # 2) FeedForward -> Residual & Norm
```

### 5. `Encoder(nn.Module)`
- **역할**:  
  - 여러 개의 `EncoderLayer`를 쌓아 전체 인코더를 구성.
  - 입력 단어 임베딩, 위치 임베딩을 더한 뒤 각 레이어를 거쳐서 최종 인코더 출력을 생성.
  - 보통 소스 문장 입력을 인코더로부터 받아 **인코딩**된 정보를 디코더에 전달.
- **구성**:  
  - `word_embedding`: 단어 임베딩.
  - `position_embedding`: 위치 임베딩(여기서는 “학습 가능”한 임베딩).
  - `layers`: N개의 `EncoderLayer` 리스트.

### 6. `DecoderLayer(nn.Module)`
- **역할**:  
  하나의 디코더 레이어로, 다음 순서를 가집니다:
  1. Masked Self-Attention (미래 정보 차단) → Residual & Norm
  2. Encoder-Decoder Attention → Residual & Norm
  3. Feed Forward → Residual & Norm

```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        # 1) Masked Self-Attention
        # 2) Encoder-Decoder Attention
        # 3) FeedForward
```

### 7. `Decoder(nn.Module)`
- **역할**:  
  - 여러 개의 `DecoderLayer`를 쌓아 전체 디코더를 구성.
  - 단어·위치 임베딩을 합산한 뒤, Masked Self-Attention을 거쳐 인코더 정보(Encoder Output)와 결합.
  - 최종적으로 단어 집합 크기(`vocab_size`)만큼 로짓을 출력(분류기).

### 8. `Transformer(nn.Module)`
- **역할**:  
  - 최상위 모듈. 인코더와 디코더를 결합.
  - 소스 입력(`src`)은 인코더를 통해 `enc_out`을 얻고, 타겟 입력(`tgt`)은 디코더를 통해 최종 단어 분포를 얻음.
  - `make_src_mask`, `make_tgt_mask` 함수를 이용해 **패딩 마스킹**과 **캐주얼 마스킹**을 생성 후 적용.

```python
class Transformer(nn.Module):
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out
```

---

## 마스킹(Masking) 주의사항

- **소스 마스크(`make_src_mask`)**  
  - 패딩된 위치(토큰 ID = 0 등)에 대한 어텐션을 막기 위해 사용.
- **타겟 마스크(`make_tgt_mask`)**  
  - **캐주얼 마스킹**(Look-Ahead Mask): 디코더가 미래 토큰을 보지 못하도록 하삼각 행렬 형태( \(\text{tril}\) )로 마스킹.  
  - **패딩 마스킹**: 0인 토큰에 대한 어텐션 무효화.  
  - 두 마스킹을 **결합**해야 논문과 동일하게 동작합니다.

```python
# 예시: 패딩 토큰 0, tgt_mask에 pad_mask & causal_mask를 모두 적용
def make_tgt_mask(self, tgt: torch.Tensor):
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool().unsqueeze(0).unsqueeze(1)
    tgt_mask = pad_mask & causal_mask
    return tgt_mask
```

---

## 실행 예시

아래는 메인 코드 블록(`if __name__ == "__main__": ...`)에서 Transformer를 사용하는 예시입니다.

1. **하이퍼파라미터 설정**  
   ```python
   src_vocab_size = 10000
   tgt_vocab_size = 10000
   src_max_length = 100
   tgt_max_length = 100
   embed_size = 512
   num_heads = 8
   num_layers = 6
   hidden_dim = 2048
   dropout = 0.1
   ```

2. **모델 및 임의 데이터 생성**  
   ```python
   model = Transformer(src_vocab_size, tgt_vocab_size,
                       src_max_length, tgt_max_length,
                       embed_size, num_heads, num_layers,
                       hidden_dim, dropout)
   
   src = torch.randint(0, src_vocab_size, (2, 10))  # (batch_size=2, src_seq_len=10)
   tgt = torch.randint(0, tgt_vocab_size, (2, 10))  # (batch_size=2, tgt_seq_len=10)
   ```

3. **Forward Pass**  
   ```python
   out = model(src, tgt)
   print("Transformer 출력:", out.shape)  # (2, 10, 10000)
   ```

---

## 참고 자료

- [Attention Is All You Need (논문)](https://arxiv.org/abs/1706.03762)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)

---

## 라이선스

이 코드는 자유롭게 수정 및 배포 가능합니다. (원 논문의 라이선스를 따르며, 학술적 목적으로 사용하길 권장합니다.)

---

이상으로 “Attention is All You Need” 논문 기반 Transformer 코드를 간단히 살펴보았습니다.  