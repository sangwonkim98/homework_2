"""
Project: Seq2Seq-Translator
Module: train.py
Description: 모델 학습을 위한 메인 스크립트입니다.
             데이터 로딩, 모델 초기화, 학습 루프, 검증, 모델 저장 등의 과정을 포함합니다.
"""

import os, sys
import yaml
import time
import torch
import pickle
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from utils.utils import (
    readLangs, get_dataloader, calculate_bleu, load_word2vec,
    PAD_token, SOS_token, EOS_token
)
from src.model import EncoderRNN, DecoderRNN, Seq2Seq

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader, val_pairs, input_lang, output_lang, config, args):
    """
    모델 학습 및 검증을 수행하는 메인 함수입니다.

    Args:
        model (nn.Module): 학습할 Seq2Seq 모델
        train_dataloader (DataLoader): 학습용 데이터로더
        val_pairs (list): 검증용 문장 쌍 리스트
        input_lang (Lang): 입력 언어 단어 사전
        output_lang (Lang): 출력 언어 단어 사전
        config (dict): config.yaml 파일에서 로드한 설정값
        args (argparse.Namespace): 커맨드 라인 인자
    """
    # TensorBoard 로깅 설정
    log_dir = os.path.join(ROOT_DIR, 'logs', args.exp_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log will be saved in: {log_dir}")

    # 옵티마이저 및 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # CrossEntropyLoss 사용. padding 토큰은 손실 계산에서 제외
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    start_time = time.time()
    best_val_bleu = 0.0

    # 설정된 에포크만큼 학습 반복
    for epoch in range(1, config['epochs'] + 1):
        model.train()  # 학습 모드로 설정
        total_loss = 0
        
        # 미니배치 단위로 학습 진행
        for src_batch, tgt_batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            
            optimizer.zero_grad() # 그래디언트 초기화

            # 모델 forward pass
            output = model(src_batch, tgt_batch, teacher_forcing_ratio=args.teacher_forcing)
            
            # 손실 계산을 위해 출력과 타겟의 차원을 맞춤
            # output: (B, T, V) -> (B*T, V)
            # target: (B, T) -> (B*T,)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = tgt_batch[:, 1:].contiguous().view(-1) # <sos> 토큰 제외
            
            loss = criterion(output, target)
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}/{config['epochs']}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")

        # --- 검증 --- 
        val_bleu = calculate_bleu(model, val_pairs, input_lang, output_lang)
        print(f"Validation BLEU score: {val_bleu:.4f}")
        
        # TensorBoard에 학습 손실과 검증 BLEU 점수 기록
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Validation/BLEU', val_bleu, epoch)

        # --- 모델 저장 --- 
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            checkpoint_path = os.path.join(ROOT_DIR, f'{args.exp_name}_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path} (Best BLEU: {best_val_bleu:.4f})")

    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    # --- 인자 파싱 --- 
    parser = argparse.ArgumentParser(description='Seq2Seq-Translator Training')
    parser.add_argument('--exp_name', type=str, default='default_experiment', help="Experiment name for logging and saving models")
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument('--use_word2vec', action='store_true', help='Use pre-trained Word2Vec')
    parser.add_argument('--w2v_path', type=str, default='GoogleNews-vectors-negative300.bin', help='Path to pre-trained word2vec .bin file')
    args = parser.parse_args()

    # --- 설정 및 데이터 로드 --- 
    config_path = os.path.join(ROOT_DIR, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir_abs = os.path.join(ROOT_DIR, config['data_dir'])
    input_lang, output_lang, pairs = readLangs('eng', 'fra', os.path.join(data_dir_abs, config['train_file']))
    _, _, val_pairs = readLangs('eng', 'fra', os.path.join(data_dir_abs, config['val_file']))
    
    # --- 단어 사전 구축 --- 
    # 학습, 검증 데이터에 있는 모든 단어를 단어 사전에 추가
    for pair in pairs + val_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Vocabularies built.")

    # --- 단어 사전 객체 저장 ---
    # 테스트 시에 동일한 단어 사전을 사용하기 위해 pickle로 저장
    lang_input_path = os.path.join(ROOT_DIR, f'{args.exp_name}_input_lang.pkl')
    lang_output_path = os.path.join(ROOT_DIR, f'{args.exp_name}_output_lang.pkl')
    with open(lang_input_path, 'wb') as f:
        pickle.dump(input_lang, f)
    with open(lang_output_path, 'wb') as f:
        pickle.dump(output_lang, f)
    print(f"Language objects saved to {args.exp_name}_[input/output]_lang.pkl")
    
    # --- Word2Vec 가중치 로드 --- 
    eng_w2v, fra_w2v = None, None
    if args.use_word2vec:
        w2v_abs_path = os.path.join(ROOT_DIR, args.w2v_path)
        if not os.path.exists(w2v_abs_path):
            print(f"Error: Word2Vec file not found at {w2v_abs_path}")
            print("Please decompress .gz file or check the path.")
            sys.exit(1)
        # 참고: 영어와 불어 모두 동일한 Google News Word2Vec을 사용.
        # 이는 다국어 임베딩이 아니므로 성능에 한계가 있을 수 있음.
        eng_w2v = load_word2vec(input_lang, w2v_abs_path, config['embedding_dim'])
        fra_w2v = load_word2vec(output_lang, w2v_abs_path, config['embedding_dim'])
        print("Word2Vec loaded for both languages.")

    # --- 데이터로더 생성 ---
    batch_size = config.get('batch_size', 256)
    train_dataloader = get_dataloader(pairs, input_lang, output_lang, batch_size)
    
    print("-" * 50)

    # --- 모델 초기화 ---
    encoder = EncoderRNN(
        vocab=input_lang.n_words, emb_dim=config['embedding_dim'],
        hid=config['hidden_size'], pad_id=PAD_token, pretrained_vec=eng_w2v
    ).to(device)
    decoder = DecoderRNN(
        vocab=output_lang.n_words, emb_dim=config['embedding_dim'],
        hid=config['hidden_size'], pad_id=PAD_token, pretrained_vec=fra_w2v
    ).to(device)
    model = Seq2Seq(encoder, decoder, sos_id=SOS_token, eos_id=EOS_token).to(device)
    
    print(f"Running experiment: {args.exp_name}")
    print(f"Device: {device}")
    print("-" * 50)

    # --- 학습 시작 ---
    train(model, train_dataloader, val_pairs, input_lang, output_lang, config, args)
