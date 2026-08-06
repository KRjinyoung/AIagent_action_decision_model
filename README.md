# AIagent_action_decision_model

Dacon 코드 제출 대회 — AI 코딩 에이전트가 사용자와 대화하는 도중, 특정 시점에서 다음에 취할 행동(action)을 예측하는 모델입니다. 세션 메타정보·작업공간 상태·대화 이력·현재 사용자 발화를 입력으로 받아 14개 행동 클래스 중 하나를 예측합니다.

## 문제 정의

- **입력**: `session_meta`(요금제, 언어 선호, 잔여 토큰, 턴 번호, 경과 시간, 작업공간 상태), `history`(직전까지의 대화·행동 기록, 0~12개), `current_prompt`(현재 사용자 발화)
- **출력**: 아래 14개 클래스 중 하나

| 그룹 | 성격 | 클래스 |
| --- | --- | --- |
| group1 | 탐색/조회 | `read_file`, `grep_search`, `glob_pattern`, `list_directory` |
| group2 | 판단/확장 | `plan_task`, `ask_user`, `web_search` |
| group3 | 실행/검증 | `run_bash`, `run_tests`, `lint_or_typecheck` |
| group4 | 수정/생성 | `edit_file`, `write_file`, `apply_patch` |
| group5 | 응답만 | `respond_only` |

- **평가**: macro F1
- **제출 방식**: 코드 제출 대회. `model/`, `script.py`, `requirements.txt`로 구성된 `submit.zip`을 제출하면 평가 서버가 `script.py`를 실행해 `output/submission.csv`를 생성합니다. (자세한 제약은 [규칙.txt](규칙.txt), [명세서.txt](명세서.txt) 참고)

## 아키텍처: 라우터 + 그룹별 전문가 모델

먼저 라우터가 각 샘플을 5개 그룹 중 하나로 분류하고, 그룹별로 특화된 전문가 모델이 최종 14-클래스 예측을 수행하는 2단계 구조입니다.

```
test.jsonl
    │
    ▼
router_model.pkl  ──► group1 | group2 | group3 | group4 | group5
    │                    │        │        │        │        │
    ▼                    ▼        ▼        ▼        ▼        ▼
router_features.py   g1_predict g2_predict g3_predict g4(RoBERTa)  respond_only
                      (탐색)     (판단)     (실행)     (수정/생성)   (고정)
    │                    │        │        │        │        │
    └────────────────────┴────────┴────────┴────────┴────────┘
                              ▼
                     output/submission.csv
```

### 그룹별 모델 구성 (`submit_final/`)

- **router**: 그룹 분류기 (`router_model.pkl`, `router_features.py`)
- **group1** (탐색/조회): 압축 XGBoost+ExtraTrees 베이스 + grep/read 전용 linear·RoBERTa 전문가 + 유틸리티 게이트 (`model/group1/`, `g1_feature.py`, `g1_predict.py`)
- **group2** (판단/확장): HistGradientBoosting + TF-IDF word/char LogisticRegression을 세션 GroupKFold OOF 스태커로 결합한 경량 앙상블, 언어모델 미사용 (`model/group2_model.pkl`, `g2_features.py`, `g2_predict.py`)
- **group3** (실행/검증): LightGBM+CatBoost+XGBoost 구조화 모델(V4) + Context Recovery BERT(multilingual DistilBERT)를 `source=="sim"` 레코드에 블렌드 (`model/group3_model.pkl`, `model/group3_bert/`, `g3_features.py`, `g3_predict.py`)
- **group4** (수정/생성): `klue/roberta-base` 파인튜닝, fp16 추론 (`model/group4/`, `g4_features.py`)
- **group5** (응답만): 별도 모델 없이 라우터 미매칭 시 `respond_only` 고정 출력

각 그룹 폴더(`group{1..4}_final/`, `router_final/`)는 해당 모델의 학습 코드·원본 산출물을 보관하는 소스이며, `submit_final/`이 실제로 zip에 담기는 배포본입니다.

## 저장소 구조

```
├── group1_final/       # group1 학습 코드 + 모델 원본
├── group2_final/       # group2 학습 코드 + 모델 원본
├── group3_final/       # group3 학습 코드 + 모델 원본
├── group4_final/       # group4 학습 코드 + 모델 원본 (README 포함)
├── router_final/       # 라우터 학습 코드 + 모델 원본
├── submit_final/       # 실제 제출 zip의 원본 디렉토리 (script.py 포함)
├── submit/             # 베이스라인 기반 초기 제출 구성 (참고용)
├── submit.zip          # 최신 제출 파일
├── submit_old.zip      # 이전 제출 백업
├── 명세서.txt           # 데이터/문제 정의
├── 규칙.txt             # 제출 방식 및 대회 규칙
└── 공식 추론 베이스라인.txt  # 대회 측 베이스라인 추론 코드
```

## 제출 파일(submit.zip) 빌드

`submit_final/`이 zip의 원본입니다. 각 `group{N}_final/`, `router_final/`에서 최종 features/model을 `submit_final/`로 복사해 구성하며, `submit_final/script.py`가 라우터 + 4개 그룹 추론을 통합 실행합니다.

zip 생성은 반드시 Python `zipfile`로 수행합니다 (Windows `Compress-Archive`는 경로 구분자가 `\`로 들어가 Linux 평가 서버에서 구조가 깨집니다):

```python
import os, zipfile

with zipfile.ZipFile("submit.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("submit_final"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, "submit_final").replace(os.sep, "/")
            zf.write(path, rel)
```

## 평가 서버 환경

- OS: Ubuntu 22.04.5 LTS / Python 3.11.15 / CUDA 12.8 / GPU: NVIDIA T4 (16GB)
- 사전 설치: `torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scikit-learn==1.8.0`, `transformers==4.46.3` 등 ([규칙.txt](규칙.txt) 참고)
- 제약: zip 용량 최대 1GB, 패키지 설치·추론 각 10분 이내, 인터넷 접속 불가
