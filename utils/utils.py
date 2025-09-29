"""
Project: Seq2Seq-Translator
Module: utils.py
Description: 데이터 전처리, 단어 사전 관리, 데이터로더 생성, 모델 평가 등
             프로젝트 전반에서 사용되는 유틸리티 함수들을 포함합니다.
"""

import unicodedata
import re
import torch
import numpy as np
import heapq
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from gensim.models import KeyedVectors

# 특수 토큰 정의
SOS_token = 0  # Start of Sentence
EOS_token = 1  # End of Sentence
PAD_token = 2  # Padding

# 디바이스 설정 (CUDA 사용 가능하면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    """
    언어의 단어 사전을 관리하는 클래스.
    단어와 인덱스 간의 매핑, 단어 빈도수 계산 등을 담당합니다.
    """
    def __init__(self, name):
        """
        Args:
            name (str): 언어의 이름 (e.g., 'eng', 'fra')
        """
        self.name = name
        self.word2index = {"SOS": SOS_token, "EOS": EOS_token, "<pad>": PAD_token}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", PAD_token: "<pad>"}
        self.n_words = 3  # SOS, EOS, PAD 포함

    def addSentence(self, sentence):
        """문장을 단어 단위로 분리하여 단어 사전에 추가합니다."""
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """단어를 단어 사전에 추가합니다."""
        if word not in self.word2index:
            # 새로운 단어일 경우
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            # 이미 있는 단어일 경우
            self.word2count[word] += 1

def load_word2vec(lang, w2v_path, emb_dim):
    """
    사전 학습된 Word2Vec 모델을 로드하여 임베딩 행렬을 생성합니다.
    단어 사전에 있는 단어가 Word2Vec 모델에 존재하면 해당 벡터를 사용하고,
    없으면 랜덤 벡터로 초기화합니다.

    Args:
        lang (Lang): 단어 사전 객체
        w2v_path (str): Word2Vec 모델 파일 경로
        emb_dim (int): 임베딩 차원

    Returns:
        torch.FloatTensor: 생성된 임베딩 행렬
    """
    print(f"Loading Word2Vec model from {w2v_path}...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    # 임베딩 행렬을 랜덤 값으로 초기화
    embedding_matrix = np.random.rand(lang.n_words, emb_dim)
    
    hits = 0
    misses = 0
    for word, i in lang.word2index.items():
        if word in w2v_model:
            # 단어가 Word2Vec 모델에 있으면 해당 벡터로 교체
            embedding_matrix[i] = w2v_model[word]
            hits += 1
        else:
            # 없으면 랜덤 초기화 값 유지
            misses += 1
            
    print(f"Converted {hits} words ({misses} misses)")
    return torch.FloatTensor(embedding_matrix)

def normalizeString(s):
    """
    문자열을 소문자로 변환하고, 유니코드 문자를 정규화하며,
    특수문자 주변에 공백을 추가하는 등 텍스트를 정제합니다.

    Args:
        s (str): 정규화할 문자열

    Returns:
        str: 정규화된 문자열
    """
    s = s.lower().strip()
    # 유니코드 문자 정규화 (e.g., è -> e)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # 구두점 주변에 공백 추가
    s = re.sub(r"([.!?])", r" \1", s)
    # 알파벳과 일부 특수문자를 제외한 나머지 문자들은 공백으로 대체
    s = re.sub(r"[^a-zA-Zàáâäçéèêëïîôöùúûüÿæœ.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, file_path):
    """
    데이터 파일을 읽어 영어-프랑스어 문장 쌍을 만들고,
    각 언어에 대한 Lang 객체를 생성합니다.

    Args:
        lang1 (str): 입력 언어 이름
        lang2 (str): 출력 언어 이름
        file_path (str): 데이터 파일 경로

    Returns:
        tuple: (input_lang, output_lang, pairs)
    """
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def tensorFromSentence(lang, sentence):
    """
    문장을 단어 인덱스의 텐서로 변환합니다. 문장 끝에 EOS 토큰을 추가합니다.

    Args:
        lang (Lang): 해당 언어의 단어 사전 객체
        sentence (str): 텐서로 변환할 문장

    Returns:
        torch.Tensor: 단어 인덱스로 구성된 텐서
    """
    indexes = [lang.word2index.get(word, PAD_token) for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device)

class TranslationDataset(Dataset):
    """
    PyTorch Dataset 클래스. 문장 쌍을 텐서로 변환하여 제공합니다.
    """
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """인덱스에 해당하는 문장 쌍을 텐서로 변환하여 반환합니다."""
        input_sentence = self.pairs[idx][0]
        output_sentence = self.pairs[idx][1]
        input_tensor = tensorFromSentence(self.input_lang, input_sentence)
        output_tensor = tensorFromSentence(self.output_lang, output_sentence)
        return input_tensor, output_tensor

def collate_fn(batch):
    """
    DataLoader에서 사용할 collate 함수.
    배치 내의 문장들을 동일한 길이로 만들기 위해 패딩을 추가합니다.

    Args:
        batch (list): (입력 텐서, 출력 텐서) 쌍의 리스트

    Returns:
        tuple: 패딩된 (입력 배치 텐서, 출력 배치 텐서)
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    # pad_sequence를 사용하여 배치 내의 시퀀스들을 가장 긴 시퀀스 길이에 맞춰 패딩
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_token)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_token)
    return src_padded, tgt_padded

def get_dataloader(pairs, input_lang, output_lang, batch_size):
    """
    Dataset과 DataLoader를 생성하여 반환합니다.

    Args:
        pairs (list): 문장 쌍 리스트
        input_lang (Lang): 입력 언어 단어 사전
        output_lang (Lang): 출력 언어 단어 사전
        batch_size (int): 배치 크기

    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# -----------------------
# 추론(번역) 및 평가 함수
# -----------------------

def evaluate(model, input_lang, output_lang, sentence, search_method='greedy', beam_size=3, max_length=20):
    """
    주어진 문장을 번역하는 함수. 탐색 방법에 따라 greedy 또는 beam search를 사용합니다.

    Args:
        model (nn.Module): 학습된 Seq2Seq 모델
        input_lang (Lang): 입력 언어 단어 사전
        output_lang (Lang): 출력 언어 단어 사전
        sentence (str): 번역할 문장
        search_method (str): 'greedy' 또는 'beam'
        beam_size (int): 빔 서치에 사용할 빔 크기
        max_length (int): 생성할 문장의 최대 길이

    Returns:
        str: 번역된 문장
    """
    if search_method == 'greedy':
        return evaluate_greedy(model, input_lang, output_lang, sentence, max_length)
    elif search_method == 'beam':
        return evaluate_beam(model, input_lang, output_lang, sentence, beam_size, max_length)
    else:
        raise ValueError("Search method not supported")

def evaluate_greedy(model, input_lang, output_lang, sentence, max_length=20):
    """Greedy search를 사용하여 문장을 번역합니다."""
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).unsqueeze(0)
        enc_out, state = model.encoder(input_tensor)
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoded_words = []

        for _ in range(max_length):
            # 디코더를 한 스텝씩 실행
            logits, state = model.decoder.step(decoder_input.squeeze(1), state)
            # 가장 확률이 높은 단어 선택
            topi = logits.argmax(1)
            
            if topi.item() == EOS_token:
                # EOS 토큰이 나오면 번역 종료
                break
            
            word = output_lang.index2word.get(topi.item(), "<unk>")
            decoded_words.append(word)
            # 예측된 단어를 다음 스텝의 입력으로 사용
            decoder_input = topi.unsqueeze(1)
            
        return " ".join(decoded_words)

def evaluate_beam(model, input_lang, output_lang, sentence, beam_size=3, max_length=20):
    """Beam search를 사용하여 문장을 번역합니다."""
    model.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).unsqueeze(0)
        enc_out, state = model.encoder(input_tensor)

        # 빔: (누적 로그 확률, 현재까지의 시퀀스, 디코더 상태)
        start_node = (0, [SOS_token], state)
        beams = [start_node]
        
        for _ in range(max_length):
            new_beams = []
            for log_prob, sequence, current_state in beams:
                if sequence[-1] == EOS_token:
                    # 이미 끝난 시퀀스는 그대로 유지
                    new_beams.append((log_prob, sequence, current_state))
                    continue

                last_word = torch.tensor([sequence[-1]], device=device)
                logits, next_state = model.decoder.step(last_word, current_state)
                log_probs = torch.log_softmax(logits, dim=1)
                
                # 다음 단어 후보군(beam_size개)을 탐색
                top_log_probs, top_indices = log_probs.topk(beam_size)

                for i in range(beam_size):
                    next_word_idx = top_indices[0][i].item()
                    new_log_prob = log_prob + top_log_probs[0][i].item()
                    new_sequence = sequence + [next_word_idx]
                    new_beams.append((new_log_prob, new_sequence, next_state))
            
            # 생성된 후보 빔들 중에서 가장 확률이 높은 beam_size개만 선택
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

            if all(b[1][-1] == EOS_token for b in beams):
                # 모든 빔이 EOS로 끝나면 탐색 종료
                break

        # 가장 확률이 높은 빔의 시퀀스를 최종 결과로 선택
        best_sequence = beams[0][1]
        decoded_words = [output_lang.index2word.get(idx, "<unk>") for idx in best_sequence if idx not in [SOS_token, EOS_token]]
        return " ".join(decoded_words)

def calculate_bleu(model, pairs, input_lang, output_lang, search_method='greedy', beam_size=3, max_length=20):
    """
    주어진 데이터셋에 대해 BLEU 점수를 계산합니다.

    Args:
        model (nn.Module): 평가할 모델
        pairs (list): 평가할 문장 쌍 리스트
        input_lang (Lang): 입력 언어 단어 사전
        output_lang (Lang): 출력 언어 단어 사전
        search_method (str): 'greedy' 또는 'beam'
        beam_size (int): 빔 서치 크기
        max_length (int): 최대 생성 길이

    Returns:
        float: 계산된 BLEU 점수
    """
    model.eval()
    references = []  # 실제 정답 문장들
    hypotheses = []  # 모델이 예측한 문장들
    chencherry = SmoothingFunction()

    for pair in tqdm(pairs, desc=f"Calculating BLEU ({search_method})"):
        input_sentence = pair[0]
        target_sentence = pair[1]
        
        predicted_sentence = evaluate(model, input_lang, output_lang, input_sentence, search_method, beam_size, max_length)
        
        references.append([target_sentence.split()])
        hypotheses.append(predicted_sentence.split())
        
    # NLTK 라이브러리를 사용하여 BLEU 점수 계산
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)
    return bleu_score