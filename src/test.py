"""
Project: Seq2Seq-Translator
Module: test.py
Description: 학습된 모델의 성능을 평가하는 스크립트입니다.
             저장된 모델 가중치와 단어 사전을 불러와 테스트 데이터셋에 대한
             번역 예시를 출력하고, 전체 BLEU 점수를 계산합니다.
"""

import os, sys
import yaml
import torch
import random
import argparse
import pickle

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from utils.utils import (
    readLangs, evaluate, calculate_bleu, PAD_token, SOS_token, EOS_token
)
from src.model import EncoderRNN, DecoderRNN, Seq2Seq

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # --- 인자 파싱 ---
    parser = argparse.ArgumentParser(description='Seq2Seq-Translator Testing')
    parser.add_argument('--exp_name', type=str, default='default_experiment', help='Experiment name to load the best model and vocabs')
    parser.add_argument('--search_method', type=str, default='greedy', choices=['greedy', 'beam'], help='Search method for decoding')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search')
    args = parser.parse_args()

    # --- 설정 파일 로드 ---
    config_path = os.path.join(ROOT_DIR, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 단어 사전 및 테스트 데이터 로드 ---
    data_dir_abs = os.path.join(ROOT_DIR, config['data_dir'])
    
    # 학습 시 저장된 단어 사전(Lang) 객체를 불러옴
    lang_input_path = os.path.join(ROOT_DIR, f'{args.exp_name}_input_lang.pkl')
    lang_output_path = os.path.join(ROOT_DIR, f'{args.exp_name}_output_lang.pkl')

    try:
        with open(lang_input_path, 'rb') as f:
            input_lang = pickle.load(f)
        with open(lang_output_path, 'rb') as f:
            output_lang = pickle.load(f)
        print("Language objects loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Language files not found for experiment '{args.exp_name}'.")
        print("Please run the training script first to generate language files.")
        sys.exit(1)

    # 테스트 데이터만 로드
    # readLangs는 새로운 Lang 객체를 반환하지만, 여기서는 test_pairs만 사용하고 Lang 객체는 무시
    _, _, test_pairs = readLangs(input_lang.name, output_lang.name, os.path.join(data_dir_abs, config['test_file']))
    print("Test data loaded.")

    # --- 모델 초기화 및 가중치 로드 ---
    # 모델 구조는 학습 때와 동일하게 정의
    # 사전 학습된 Word2Vec은 임베딩 레이어의 초기 가중치로만 사용되었으므로,
    # 여기서는 fine-tuning된 가중치를 불러오기 때문에 pretrained_vec=None으로 설정
    encoder = EncoderRNN(
        vocab=input_lang.n_words, emb_dim=config['embedding_dim'],
        hid=config['hidden_size'], pad_id=PAD_token
    ).to(device)
    decoder = DecoderRNN(
        vocab=output_lang.n_words, emb_dim=config['embedding_dim'],
        hid=config['hidden_size'], pad_id=PAD_token
    ).to(device)
    model = Seq2Seq(encoder, decoder, sos_id=SOS_token, eos_id=EOS_token).to(device)

    # 학습을 통해 저장된 모델 가중치(state_dict)를 불러옴
    checkpoint_path = os.path.join(ROOT_DIR, f'{args.exp_name}_best.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Error: No checkpoint found at {checkpoint_path}")
        sys.exit(1)
    
    model.eval() # 모델을 평가 모드로 설정
    print("-" * 50)

    # --- 랜덤 샘플 번역 테스트 ---
    print(f"Testing with random samples (Search: {args.search_method}, Beam: {args.beam_size if args.search_method == 'beam' else 'N/A'}):")
    for _ in range(5):
        pair = random.choice(test_pairs)
        predicted_translation = evaluate(model, input_lang, output_lang, pair[0], args.search_method, args.beam_size)
        print(f"> Input:    {pair[0]}")
        print(f"= Target:   {pair[1]}")
        print(f"< Predict:  {predicted_translation}")
        print("")
    
    print("-" * 50)

    # --- 전체 테스트셋에 대한 BLEU 점수 계산 ---
    bleu_score = calculate_bleu(model, test_pairs, input_lang, output_lang, args.search_method, args.beam_size)
    print(f"Total BLEU score on the test set: {bleu_score:.4f}")