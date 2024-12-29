import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# MultiHeadAttention 클래스
class MultiHeadAttention(nn.Module):
    """
    논문 "Attention is All You Need"의 Multi-Head Attention 메커니즘을 구현한 클래스.
    이 클래스는 입력 쿼리(Query), 키(Key), 값(Value)에 대해 점곱(Sdot-Product Attention)을 수행하며,
    여러 헤드(Multiple Heads)를 병렬적으로 사용해 정보 표현력을 확장.
    """
    def __init__(self, embed_size: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size  # 입력 임베딩 크기. 논문에서는 d_model로 표현.
        self.num_heads = num_heads  # 병렬적인 헤드의 수. 논문에서 h로 표현.
        self.head_dim = embed_size // num_heads  # 각 헤드의 차원 크기. 논문에서 d_k = d_model / h.

        # d_model이 h로 정확히 나누어떨어지는지 확인. 그렇지 않으면 에러 발생.
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"

        # 쿼리(Q), 키(K), 값(V)를 생성하기 위한 선형 변환. 논문에서 W_Q, W_K, W_V에 해당.
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        # 멀티헤드 출력을 하나로 병합하기 위한 선형 변환. 논문에서 W_O에 해당.
        self.fc_out = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        논문의 Scaled Dot-Product Attention 부분을 구현.
        쿼리와 키 간 점곱 후, 키의 차원 크기(d_k)의 제곱근으로 스케일링.
        """
        d_k = query.size(-1)  # 키의 차원 크기(d_k).
        # 쿼리와 키의 점곱 계산. 논문에서 Q * K^T.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 스케일링 수행.

        if mask is not None:
            # 마스크가 주어진 경우, 패딩된 부분 또는 미래 토큰을 무한대로 낮은 값으로 설정.
            scores = scores.masked_fill(mask == 0, float('-1e20'))

        # 점수에 소프트맥스를 적용하여 Attention Weight 계산. 논문에서 softmax(Q * K^T / sqrt(d_k)).
        attention = F.softmax(scores, dim=-1)
        # Attention Weight와 Value의 점곱으로 최종 출력을 계산. 논문에서 softmax(...) * V.
        out = torch.matmul(attention, value)
        return out, attention

    def forward(
        self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        논문의 Multi-Head Attention 모듈 전체를 구현.
        각 입력에 대해 쿼리(Q), 키(K), 값(V)를 계산하고, 이를 여러 헤드로 나누어 병렬 계산 후 합침.
        """
        N = query.shape[0]  # 배치 크기 (batch size).
        
        # 쿼리(Q), 키(K), 값(V)를 선형 변환 후, 각 헤드로 분리.
        value = self.values(value).view(N, -1, self.num_heads, self.head_dim)
        key = self.keys(key).view(N, -1, self.num_heads, self.head_dim)
        query = self.queries(query).view(N, -1, self.num_heads, self.head_dim)

        # 헤드 차원을 첫 번째 축으로 변환. (배치 크기, 헤드 수, 시퀀스 길이, 각 헤드의 차원).
        value = value.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        # 스케일 점곱 주의력 메커니즘 수행.
        out, _ = self.scaled_dot_product_attention(query, key, value, mask)

        # 각 헤드의 출력을 병합하여 원래의 차원으로 변환.
        out = out.permute(0, 2, 1, 3).contiguous()  # (배치 크기, 시퀀스 길이, 임베딩 크기).
        out = out.view(N, -1, self.embed_size)
        return self.fc_out(out)  # 최종 선형 변환 적용.

# FeedForward 클래스
class FeedForward(nn.Module):
    """
    논문 "Attention is All You Need"에서 Position-wise Feed Forward Network 부분을 구현한 클래스.
    각 시퀀스의 위치에 대해 독립적으로 작동하며, 두 개의 선형 변환과 ReLU 활성화 함수를 포함.
    """
    def __init__(self, embed_size: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)  # 첫 번째 선형 변환. (논문에서 W_1).
        self.fc2 = nn.Linear(hidden_dim, embed_size)  # 두 번째 선형 변환. (논문에서 W_2).
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 추가.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        논문의 Feed Forward 부분:
        max(0, xW_1 + b_1)W_2 + b_2.
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))  # ReLU 적용 후 두 번째 선형 변환.

# LayerNormalization 클래스
class LayerNormalization(nn.Module):
    """
    논문에서 사용된 Residual Connection 이후의 Layer Normalization을 구현.
    평균과 분산을 이용해 입력을 정규화하고, 스케일과 이동 매개변수를 학습.
    """
    def __init__(self, embed_size: int, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.gain = nn.Parameter(torch.ones(embed_size))  # 정규화된 값에 곱할 스케일 매개변수.
        self.bias = nn.Parameter(torch.zeros(embed_size))  # 정규화된 값에 더할 이동 매개변수.
        self.eps = eps  # 안정성을 위해 추가된 작은 값.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 x를 평균과 표준편차를 기반으로 정규화:
        (x - mean) / (std + eps).
        """
        mean = x.mean(dim=-1, keepdim=True)  # 마지막 차원에서 평균 계산.
        std = x.std(dim=-1, keepdim=True)  # 마지막 차원에서 표준편차 계산.
        return self.gain * (x - mean) / (std + self.eps) + self.bias  # 정규화 후 스케일링 및 이동.

# EncoderLayer 클래스
class EncoderLayer(nn.Module):
    """
    논문 "Attention is All You Need"에서 인코더 레이어를 구현.
    각 레이어는 Multi-Head Attention, Feed Forward, Residual Connection, Layer Normalization으로 구성.
    """
    def __init__(self, embed_size: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)  # 멀티헤드 어텐션 모듈 (논문 3.2절).
        self.norm1 = LayerNormalization(embed_size)  # 첫 번째 Layer Normalization (Residual 이후).
        self.norm2 = LayerNormalization(embed_size)  # 두 번째 Layer Normalization (Residual 이후).
        self.feed_forward = FeedForward(embed_size, hidden_dim, dropout)  # Feed Forward Network (논문 3.3절).
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 추가 (논문에서 Regularization).

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        인코더 레이어의 순차적인 연산:
        1. Multi-Head Attention → Residual Connection → Layer Normalization.
        2. Feed Forward → Residual Connection → Layer Normalization.
        """
        attn_out = self.attention(x, x, x, mask)  # Self-Attention 적용.
        x = self.norm1(x + self.dropout(attn_out))  # Residual Connection 후 Layer Normalization.
        ff_out = self.feed_forward(x)  # Position-wise Feed Forward Network 적용.
        x = self.norm2(x + self.dropout(ff_out))  # Residual Connection 후 Layer Normalization.
        return x

# Encoder 클래스
class Encoder(nn.Module):
    """
    논문 "Attention is All You Need"에서 인코더 전체 구조를 구현.
    단어와 위치 임베딩, 여러 개의 인코더 레이어로 구성.
    """
    def __init__(
        self, 
        num_layers: int, 
        embed_size: int, 
        num_heads: int, 
        hidden_dim: int, 
        vocab_size: int, 
        max_length: int, 
        dropout: float = 0.1
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size  # 입력 임베딩 크기.
        self.word_embedding = nn.Embedding(vocab_size, embed_size)  # 단어 임베딩 (논문 3.4절).
        self.position_embedding = nn.Embedding(max_length, embed_size)  # 위치 임베딩 (논문 3.4절).
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )  # 여러 개의 인코더 레이어 (논문 3.1절).
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 추가.

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        인코더 연산:
        1. 단어 임베딩 + 위치 임베딩 → 드롭아웃.
        2. 각 인코더 레이어를 순차적으로 통과.
        """
        N, seq_length = x.shape  # 배치 크기와 시퀀스 길이.
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)  # 위치 인덱스 생성.
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  # 임베딩 합산 후 드롭아웃.
        for layer in self.layers:  # 각 인코더 레이어를 순차적으로 통과.
            out = layer(out, mask)
        return out  # 인코더 출력 (논문 3.1절).

# DecoderLayer 클래스
class DecoderLayer(nn.Module):
    """
    논문 "Attention is All You Need"에서 디코더 레이어를 구현.
    각 레이어는 Masked Multi-Head Attention, Encoder-Decoder Attention, Feed Forward로 구성.
    """
    def __init__(self, embed_size: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_size, num_heads)  # Masked Self-Attention (논문 3.2절).
        self.norm1 = LayerNormalization(embed_size)  # 첫 번째 Layer Normalization.
        self.encoder_attention = MultiHeadAttention(embed_size, num_heads)  # Encoder-Decoder Attention (논문 3.2절).
        self.norm2 = LayerNormalization(embed_size)  # 두 번째 Layer Normalization.
        self.feed_forward = FeedForward(embed_size, hidden_dim, dropout)  # Feed Forward Network (논문 3.3절).
        self.norm3 = LayerNormalization(embed_size)  # 세 번째 Layer Normalization.
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 추가.

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_out: torch.Tensor, 
        src_mask: Optional[torch.Tensor], 
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        디코더 레이어의 순차적인 연산:
        1. Masked Multi-Head Attention → Residual Connection → Layer Normalization.
        2. Encoder-Decoder Attention → Residual Connection → Layer Normalization.
        3. Feed Forward → Residual Connection → Layer Normalization.
        """
        # Masked Self-Attention: 미래 정보 차단.
        masked_attn_out = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(masked_attn_out))  # Residual Connection 후 Layer Normalization.
        
        # Encoder-Decoder Attention: 인코더의 출력 정보 활용.
        enc_dec_attn_out = self.encoder_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_out))  # Residual Connection 후 Layer Normalization.
        
        # Feed Forward Network.
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))  # Residual Connection 후 Layer Normalization.
        return x

# Decoder 클래스
class Decoder(nn.Module):
    """
    논문 "Attention is All You Need"에서 디코더 전체 구조를 구현.
    단어와 위치 임베딩, 여러 개의 디코더 레이어로 구성.
    """
    def __init__(
        self, 
        num_layers: int, 
        embed_size: int, 
        num_heads: int, 
        hidden_dim: int, 
        vocab_size: int, 
        max_length: int, 
        dropout: float = 0.1
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size  # 입력 임베딩 크기.
        self.word_embedding = nn.Embedding(vocab_size, embed_size)  # 단어 임베딩 (논문 3.4절).
        self.position_embedding = nn.Embedding(max_length, embed_size)  # 위치 임베딩 (논문 3.4절).
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )  # 여러 개의 디코더 레이어 (논문 3.1절).
        self.fc_out = nn.Linear(embed_size, vocab_size)  # 최종 출력 선형 변환 (논문 3.4절).
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 추가.

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_out: torch.Tensor, 
        src_mask: Optional[torch.Tensor], 
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        디코더 연산:
        1. 단어 임베딩 + 위치 임베딩 → 드롭아웃.
        2. 각 디코더 레이어를 순차적으로 통과.
        3. 최종 선형 변환을 통해 단어 확률 분포 출력.
        """
        N, seq_length = x.shape  # 배치 크기와 시퀀스 길이.
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)  # 위치 인덱스 생성.
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  # 임베딩 합산 후 드롭아웃.
        for layer in self.layers:  # 각 디코더 레이어를 순차적으로 통과.
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.fc_out(x)  # 단어 집합 크기로 출력 변환 (논문 3.4절).

# Transformer 클래스
class Transformer(nn.Module):
    """
    논문 "Attention is All You Need"에서 Transformer 전체 구조를 구현.
    인코더-디코더 구조로, 소스와 타겟 시퀀스를 입력받아 출력 시퀀스를 생성.
    """
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_max_length: int, 
        tgt_max_length: int, 
        embed_size: int, 
        num_heads: int, 
        num_layers: int, 
        hidden_dim: int, 
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, embed_size, num_heads, hidden_dim, src_vocab_size, src_max_length, dropout)  # 인코더 모듈.
        self.decoder = Decoder(num_layers, embed_size, num_heads, hidden_dim, tgt_vocab_size, tgt_max_length, dropout)  # 디코더 모듈.

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        소스 마스크 생성: 패딩 토큰을 고려하여 마스크 생성.
        (논문에서는 패딩된 부분에 대한 어텐션 차단.)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len).
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        타겟 마스크 생성: 패딩 마스크와 미래 정보 차단(캐주얼 마스킹)을 결합.
        """
        # (1) Pad Mask
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        
        # (2) Subsequent Mask (Causal)
        tgt_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_seq_len, tgt_seq_len)
        
        # (3) Combine
        tgt_mask = pad_mask & causal_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Transformer의 순차적인 연산:
        1. 인코더에 소스 입력과 마스크를 전달해 인코더 출력 생성.
        2. 디코더에 타겟 입력, 인코더 출력, 마스크를 전달해 최종 출력 생성.
        """
        src_mask = self.make_src_mask(src)  # 소스 마스크 생성.
        tgt_mask = self.make_tgt_mask(tgt)  # 타겟 마스크 생성.
        enc_out = self.encoder(src, src_mask)  # 인코더 출력 생성.
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)  # 디코더 출력 생성.
        return out  # 최종 출력.


#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#


if __name__ == "__main__":
    # 하이퍼파라미터
    src_vocab_size = 10000  # 소스 단어 집합 크기
    tgt_vocab_size = 10000  # 타겟 단어 집합 크기
    src_max_length = 100  # 소스 최대 문장 길이
    tgt_max_length = 100  # 타겟 최대 문장 길이
    embed_size = 512  # 임베딩 크기
    num_heads = 8  # 헤드 수
    num_layers = 6  # Encoder, Decoder 레이어 수
    hidden_dim = 2048  # FFN의 히든 크기
    dropout = 0.1  # 드롭아웃 비율

    # 입력 데이터
    src = torch.randint(0, src_vocab_size, (2, 10))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (2, 10))  # (batch_size, tgt_seq_len)

    # Transformer 모델 생성
    model = Transformer(src_vocab_size, tgt_vocab_size, src_max_length, tgt_max_length,
                        embed_size, num_heads, num_layers, hidden_dim, dropout)
    
    # Transformer 실행
    out = model(src, tgt)
    print("Transformer 출력:", out.shape)  # 출력 크기: (batch_size, tgt_seq_len, tgt_vocab_size)
