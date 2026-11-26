# 한·영 중복 질문 판별 · Siamese LSTM

두 질문이 같은 답을 요구하는지 예측하는 자연어처리 미니 프로젝트다. 영어 전용 MaLSTM을 한국어·영어 질문에 적용하기 위해 두 언어의 정렬 fastText 임베딩을 도입하고, 같은 다언어 모델에서 Attention의 효과를 측정했다. 중복의 레이블은 1, 비중복은 0이다.

## 실험 결과

같은 테스트 질문쌍에서 중복·비중복 클래스의 F1을 각각 계산하고 두 값의 평균(Macro-F1)을 구했다. 마지막 행은 세 언어 조합의 Macro-F1에 같은 가중치를 준 평균이다.

| 테스트셋 | A · 영어 기준 모델 | B · 다언어 LSTM | C · 다언어 LSTM + Attention |
|---|---:|---:|---:|
| 영–영 700쌍 | 0.6884 | **0.7258** | 0.6924 |
| 한–한 756쌍 | — | **0.7745** | 0.7567 |
| 영–한 700쌍 | — | 0.6164 | **0.6345** |
| **세 조합 평균** | — | **0.7056** | 0.6945 |

Attention을 추가하면 영–한에서 실제 중복 469쌍 중 찾아낸 건수가 294쌍에서 359쌍으로 늘었다. 비중복을 중복으로 분류한 건수도 82쌍에서 115쌍으로 늘었다. C와 B의 평균 Macro-F1 차이는 **−0.0110**이었고, 대응 부트스트랩 95% 신뢰구간은 **[−0.0316, +0.0091]**이었다. 시드 42에서 학습한 두 모델의 테스트 예측을 기준으로 전체 성능이 향상됐다고 판단하기는 어렵다.

[평가 점수와 혼동행렬](results/test_scores.csv) · [모델별 질문쌍 예측](results/)

## 모델과 데이터

| 폴더 | 역할 |
|---|---|
| [original/](original/) | 참고한 likejazz/Siamese-LSTM 원본 구현 |
| [original_fixed/](original_fixed/) | **A** · 원본의 영어 Word2Vec, 공유 LSTM, 맨해튼 유사도를 유지하고 고정 분할·가중치 저장·F1 평가를 적용 |
| [improved/](improved/) | **B** · 한국어 전처리와 정렬 fastText를 적용하고 각 질문의 마지막 유효 LSTM 은닉 상태를 비교 |
| [improved_attention/](improved_attention/) | **C** · 같은 LSTM의 모든 유효 시점에 Attention 가중치를 매겨 질문 벡터를 만드는 모델 |
| [dataset/](dataset/) | 언어 조합별 학습·검증·테스트 질문쌍과 표본 출처 |
| [results/](results/) | 세 모델의 테스트 예측, F1 집계, 성능 차이 재계산 코드 |

모델 A는 영어 7,000쌍으로 학습했다. B와 C는 동일한 분할 목록으로 중복을 정리한 20,125쌍(영–영 7,000, 한–한 6,127, 영–한 6,998)을 학습했다. 영–영과 영–한은 서로 다른 Quora 질문쌍이며, 영–한은 두 번째 영어 질문을 Argos Translate로 한국어로 번역한 자료다. [데이터 구성과 출처](dataset/README.md) · [분할 목록](improved/split_manifest.json)

세 모델은 유사도 `exp(-sum(abs(h1 - h2)))`에 판정 임계값을 적용한다. 학습 시드는 42, 배치 크기는 2,048, 학습 기간은 50 epoch다. 검증에서 선택한 체크포인트와 임계값은 A가 49 epoch·0.46, B가 50 epoch·0.61, C가 50 epoch·0.59였다. B와 C는 학습·검증 데이터, 임베딩, 모델 선택 규칙을 공유한다.

## 사전학습 임베딩

다음 파일을 프로젝트 루트의 `embedding/`에 원래 파일명으로 둔다. 세 파일의 총 용량은 약 9.32 GB다. B와 C는 같은 영어·한국어 파일을 공유하고, 기본 경로에 빠진 fastText 파일은 학습 스크립트가 내려받는다.

| 파일 | 사용하는 모델 | 출처 및 다운로드 |
|---|---|---|
| `GoogleNews-vectors-negative300.bin.gz` | A | [Google word2vec 프로젝트](https://code.google.com/archive/p/word2vec/) · [Google News 벡터 다운로드](https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz) |
| `wiki.en.align.vec` | B, C | [fastText 정렬 벡터](https://fasttext.cc/docs/en/aligned-vectors.html) · [영어 벡터 다운로드](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec) |
| `wiki.ko.align.vec` | B, C | [fastText 정렬 벡터](https://fasttext.cc/docs/en/aligned-vectors.html) · [한국어 벡터 다운로드](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ko.align.vec) |

## 실행

**저장된 결과 재계산.** Python과 NumPy만 있으면 저장소 루트에서 아래 명령으로 질문쌍별 예측에서 Macro-F1과 95% 신뢰구간을 재현할 수 있다. 신뢰구간은 같은 질문을 공유하는 쌍끼리 묶고, 각 언어 조합 안에서 두 모델의 예측을 함께 10,000회 재표집해 구한다.

```shell
python results/compare.py
```

**GPU 학습 환경.** 실험에 사용한 Windows용 Anaconda 환경은 Python 3.10, TensorFlow 2.10.1, CUDA 11.2, cuDNN 8.1로 구성했다. NLTK 영어 불용어는 모델 A에서 사용한다.

```powershell
conda env create -f original_fixed/environment.yml
conda activate malstm-gpu
python -m nltk.downloader stopwords
```

**학습과 평가.** 아래 명령은 저장소 루트 기준이다. 학습은 모델별 `artifacts/`에 가중치와 단어표를 생성하고, 평가는 그 폴더에 테스트 예측을 기록한다. 완료한 모델을 다시 학습할 때는 학습·평가 명령에 같은 새 `--run-dir`를 지정한다.

```powershell
python original_fixed/train.py
python original_fixed/evaluate.py

python improved/train.py
python improved/evaluate.py

python improved_attention/train.py
python improved_attention/evaluate.py
```

Attention 모델 학습 후에는 질문 두 개를 직접 넣어 유사도와 중복 판정을 볼 수 있다.

```powershell
python improved_attention/predict.py --languages ko en --questions "비밀번호를 어떻게 바꾸나요?" "How do I change my password?"
```

## 출처

- MaLSTM 기반 코드 · [likejazz/Siamese-LSTM](https://github.com/likejazz/Siamese-LSTM).
- 한국어 질문쌍 · [songys/Question_pair](https://github.com/songys/Question_pair).
- 영어 질문쌍 · [Quora Question Pairs · GLUE](https://huggingface.co/datasets/nyu-mll/glue/viewer/qqp).
- 정렬 임베딩 · [fastText Aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html).
