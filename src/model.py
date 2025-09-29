"""
Project: Seq2Seq-Translator
Module: model.py
Description: Seq2Seq 모델의 구성 요소인 Encoder, Decoder 및 전체 Seq2Seq 모델 클래스를 정의합니다.
"""

import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    Seq2Seq 모델의 인코더(Encoder) 부분입니다.
    입력 시퀀스를 받아 컨텍스트 벡터(Context Vector)를 생성합니다.
    """
    def __init__(self, vocab, emb_dim, hid, pad_id, pretrained_vec=None):
        """
        Args:
            vocab (int): 단어 사전의 크기
            emb_dim (int): 임베딩 벡터의 차원
            hid (int): RNN (LSTM)의 은닉 상태 차원
            pad_id (int): 패딩 토큰의 인덱스
            pretrained_vec (torch.Tensor, optional): 사전 학습된 임베딩 벡터. Defaults to None.
        """
        super().__init__()
        
        if pretrained_vec is not None:
            # 사전 학습된 벡터가 있으면, 임베딩 레이어를 해당 가중치로 초기화하고 fine-tuning
            self.emb = nn.Embedding.from_pretrained(pretrained_vec, freeze=False, padding_idx=pad_id)
        else:
            # 없으면, 단어 사전 크기에 맞춰 랜덤으로 초기화
            self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        
        # LSTM을 RNN 셀로 사용
        self.rnn = nn.LSTM(emb_dim, hid, batch_first=True)

    def forward(self, src):
        """
        인코더의 forward pass를 정의합니다.

        Args:
            src (torch.Tensor): 입력 시퀀스 텐서 (Batch, Seq_len)

        Returns:
            tuple: (RNN의 전체 출력 시퀀스, 마지막 은닉 상태(hidden state), 마지막 셀 상태(cell state))
        """
        # src: (B, S) -> x: (B, S, E)
        x = self.emb(src)
        # out: (B, S, H), (h, c): (1, B, H)
        out, (h, c) = self.rnn(x)
        return out, (h, c)

class DecoderRNN(nn.Module):
    """
    Seq2Seq 모델의 디코더(Decoder) 부분입니다.
    인코더의 컨텍스트 벡터를 받아 출력 시퀀스를 생성합니다.
    """
    def __init__(self, vocab, emb_dim, hid, pad_id, pretrained_vec=None):
        """
        Args:
            vocab (int): 단어 사전의 크기
            emb_dim (int): 임베딩 벡터의 차원
            hid (int): RNN (LSTM)의 은닉 상태 차원
            pad_id (int): 패딩 토큰의 인덱스
            pretrained_vec (torch.Tensor, optional): 사전 학습된 임베딩 벡터. Defaults to None.
        """
        super().__init__()
        
        if pretrained_vec is not None:
            self.emb = nn.Embedding.from_pretrained(pretrained_vec, freeze=False, padding_idx=pad_id)
        else:
            self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)

        self.rnn = nn.LSTM(emb_dim, hid, batch_first=True)
        # RNN의 출력을 단어 사전 크기의 벡터로 변환하여 각 단어의 확률을 계산
        self.proj = nn.Linear(hid, vocab)

    def step(self, y_prev, state):
        """
        디코더의 한 스텝(단어 하나)을 연산합니다.
        추론(evaluation) 시에 주로 사용됩니다.

        Args:
            y_prev (torch.Tensor): 이전 스텝의 예측 단어 (Batch,)
            state (tuple): 이전 스텝의 (은닉 상태, 셀 상태)

        Returns:
            tuple: (현재 스텝의 출력 로짓, 현재 스텝의 (은닉 상태, 셀 상태))
        """
        # y_prev: (B,) -> x: (B, 1, E)
        x = self.emb(y_prev).unsqueeze(1)
        # out: (B, 1, H), state: (1, B, H)
        out, state = self.rnn(x, state)
        # logits: (B, V)
        logits = self.proj(out.squeeze(1))
        return logits, state

class Seq2Seq(nn.Module):
    """
    인코더와 디코더를 결합한 전체 Seq2Seq 모델입니다.
    """
    def __init__(self, encoder, decoder, sos_id, eos_id):
        """
        Args:
            encoder (EncoderRNN): 인코더 객체
            decoder (DecoderRNN): 디코더 객체
            sos_id (int): SOS 토큰의 인덱스
            eos_id (int): EOS 토큰의 인덱스
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        """
        Seq2Seq 모델의 forward pass를 정의합니다.
        주로 학습(training) 시에 사용됩니다.

        Args:
            src (torch.Tensor): 입력 시퀀스 텐서 (B, S_in)
            tgt (torch.Tensor): 타겟 시퀀스 텐서 (B, S_out)
            teacher_forcing_ratio (float): Teacher Forcing을 사용할 확률. Defaults to 1.0.

        Returns:
            torch.Tensor: 예측된 로짓 시퀀스 (B, S_out-1, V)
        """
        # 인코더를 통해 컨텍스트 벡터(state) 생성
        enc_out, state = self.encoder(src)
        
        B, T = tgt.size()
        # 디코더의 첫 입력은 타겟 시퀀스의 시작(<sos>) 토큰
        y = tgt[:, 0]
        
        logits_seq = []

        # 타겟 시퀀스의 길이만큼 반복 (첫 <sos> 토큰 제외)
        for t in range(1, T):
            logits, state = self.decoder.step(y, state)
            logits_seq.append(logits.unsqueeze(1))
            
            # Teacher Forcing 적용 여부 결정
            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            
            # Teacher Forcing을 사용하면 다음 입력을 실제 정답으로, 아니면 모델의 예측으로 사용
            y = tgt[:, t] if use_tf else logits.argmax(-1)

        # (B, S_out-1, V) 형태로 변환하여 반환
        return torch.cat(logits_seq, dim=1)
