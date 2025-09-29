"""
Project: Seq2Seq-Translator
Module: prepare_data.py
Description: 원본 데이터 파일(eng-fra.txt)을 읽어와
             학습, 검증, 테스트 세트로 분할하여 저장합니다.
             전체 프로세스에서 가장 먼저 1회만 실행하면 됩니다.
"""

import os
import random
from sklearn.model_selection import train_test_split

# --- 경로 설정 ---
input_file_path = 'data/eng-fra.txt'
train_file_path = 'data/train.txt'
val_file_path = 'data/val.txt'
test_file_path = 'data/test.txt'

# --- 데이터 폴더 및 원본 파일 확인 ---
if not os.path.exists('data'):
    print("Error: 'data' directory not found.")
    print("Please create the 'data' directory and place 'eng-fra.txt' inside.")
    exit()

if not os.path.exists(input_file_path):
    print(f"Error: '{input_file_path}' not found.")
    print("Please download 'eng-fra.txt' and place it in the 'data' directory.")
    exit()

# --- 데이터 읽기 ---
with open(input_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# --- 데이터 분할 (80% train, 10% validation, 10% test) ---

# 데이터 순서를 무작위로 섞음
random.shuffle(lines)

# 먼저 테스트 세트(10%)를 분리
# random_state를 고정하여 항상 동일한 분할 결과를 얻도록 함
train_val, test_data = train_test_split(lines, test_size=0.1, random_state=42)

# 남은 90% 데이터에서 검증 세트(10%)를 분리 (전체 데이터의 10%가 되도록 1/9 비율 사용)
train_data, val_data = train_test_split(train_val, test_size=1/9, random_state=42) 

def write_to_file(file_path, data):
    """리스트 형태의 데이터를 파일에 씁니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(data)

# --- 분할된 데이터를 파일로 저장 ---
write_to_file(train_file_path, train_data)
write_to_file(val_file_path, val_data)
write_to_file(test_file_path, test_data)

print("Data splitting complete.")
print(f"Training data:   {len(train_data)} lines -> {train_file_path}")
print(f"Validation data: {len(val_data)} lines -> {val_file_path}")
print(f"Test data:       {len(test_data)} lines -> {test_file_path}")
