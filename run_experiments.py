"""
Project: Seq2Seq-Translator
Module: run_experiments.py
Description: 정의된 여러 실험 설정에 따라 train.py와 test.py를 순차적으로 실행하여
             전체 실험 과정을 자동화하는 스크립트입니다.
"""

import subprocess
import os

# --- 실험 설정 매트릭스 ---
# 각 딕셔너리가 하나의 독립적인 실험을 의미합니다.
# 이 리스트에 설정을 추가하거나 수정하여 다양한 조건으로 실험을 실행할 수 있습니다.
experiments = [
    {
        "exp_name": "setting_1_word2vec_greedy",
        "use_word2vec": True,
        "teacher_forcing": 0.5,
        "search_method": "greedy",
    },
    {
        "exp_name": "setting_2_word2vec_beam",
        "use_word2vec": True,
        "teacher_forcing": 0.5,
        "search_method": "beam",
    },
    {
        "exp_name": "setting_3_no_word2vec_greedy",
        "use_word2vec": False,
        "teacher_forcing": 0.5,
        "search_method": "greedy",
    },
    {
        "exp_name": "setting_4_tf1.0_greedy",
        "use_word2vec": False,
        "teacher_forcing": 1.0, # Teacher Forcing을 항상 사용
        "search_method": "greedy",
    },
]

# 사전 학습된 Word2Vec 모델 파일 경로 (프로젝트 루트에 위치한다고 가정)
w2v_path = "GoogleNews-vectors-negative300.bin"

# 이 스크립트가 위치한 디렉토리를 기준으로 프로젝트 루트 디렉토리 설정
ROOT_DIR = os.path.dirname(__file__)

def run_command(command):
    """
    주어진 커맨드를 서브프로세스로 실행하고, 그 출력을 실시간으로 화면에 보여줍니다.

    Args:
        command (list): 실행할 커맨드와 인자들의 리스트

    Returns:
        int: 커맨드 실행 종료 코드 (0이면 성공, 그 외는 실패)
    """
    print(f"\n{'='*20}\nRunning command: {' '.join(command)}\n{'='*20}")
    # cwd=ROOT_DIR를 통해 스크립트 실행 위치를 프로젝트 루트로 보장
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT_DIR)
    
    # 실시간으로 출력 스트림 읽기
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # 최종 종료 코드 반환
    rc = process.poll()
    return rc

# --- 각 실험을 순차적으로 실행 ---
for i, config in enumerate(experiments):
    print(f"\n\n{'#'*50}")
    print(f"# Starting Experiment {i+1}/{len(experiments)}: {config['exp_name']}")
    print(f"# Config: {config}")
    print(f"{'#'*50}")

    # --- 1. 모델 학습 ---
    # train.py를 실행할 커맨드 리스트 생성
    train_command = [
        "python",
        "src/train.py",
        "--exp_name", config["exp_name"],
        "--teacher_forcing", str(config["teacher_forcing"]),
    ]
    # Word2Vec 사용 설정이 True이면 관련 인자 추가
    if config["use_word2vec"]:
        train_command.append("--use_word2vec")
        train_command.extend(["--w2v_path", w2v_path])
    
    # 학습 스크립트 실행
    train_rc = run_command(train_command)
    if train_rc != 0:
        print(f"Training failed for experiment {config['exp_name']}. Stopping.")
        break # 학습 실패 시 전체 실험 중단

    # --- 2. 모델 평가 ---
    # test.py를 실행할 커맨드 리스트 생성
    test_command = [
        "python",
        "src/test.py",
        "--exp_name", config["exp_name"],
        "--search_method", config["search_method"],
    ]
    # Beam search일 경우 beam_size 인자 추가
    if config["search_method"] == "beam":
        test_command.extend(["--beam_size", "5"]) # 빔 크기는 필요에 따라 조절

    # 평가 스크립트 실행
    test_rc = run_command(test_command)
    if test_rc != 0:
        print(f"Testing failed for experiment {config['exp_name']}. Stopping.")
        break # 평가 실패 시 전체 실험 중단

print("\nAll experiments completed.")