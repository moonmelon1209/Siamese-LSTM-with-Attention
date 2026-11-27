# 한·영 중복 질문 판별

한국어와 영어 질문 두 개가 같은 답을 요구하는지 판별하는 Siamese LSTM 프로젝트다. 영어 전용 MaLSTM을 다언어 모델로 확장한 뒤, Self-Attention을 추가하는 실험과 기본 다언어 LSTM의 학습량을 늘리는 실험을 진행했다. 중복의 레이블은 1, 비중복은 0이다.

## 모델과 테스트 결과

표의 각 점수는 중복과 비중복의 F1을 같은 비중으로 평균한 Macro-F1이다. 마지막 열은 세 언어 조합에 같은 비중을 둔 평균이다. 각 모델의 체크포인트와 판정 임계값은 검증셋에서 선택했다.

| 모델 | 학습 질문쌍 | 최대 에포크 | 영–영 (700쌍) | 한–한 (756쌍) | 영–한 (700쌍) | 세 조합 평균 |
|---|---:|---:|---:|---:|---:|---:|
| A · 영어 전용 MaLSTM | 7,000 | 50 | 0.6884 | — | — | — |
| B · 다언어 LSTM | 20,125 | 50 | 0.7258 | 0.7745 | 0.6164 | 0.7056 |
| C · B + Self-Attention | 20,125 | 50 | 0.6924 | 0.7567 | 0.6345 | 0.6945 |
| B · 학습 데이터 절반 | 10,063 | 100 | 0.7106 | 0.7502 | 0.5759 | 0.6789 |
| B · 학습 횟수 증가 | 20,125 | 100 | 0.7135 | 0.7764 | 0.6553 | 0.7151 |

**Self-Attention.** 영–한에서 중복을 맞힌 건수는 294쌍에서 359쌍으로, 비중복을 중복으로 판단한 건수는 82쌍에서 115쌍으로 늘었다. 세 조합 평균 Macro-F1의 차이(C − B)는 −0.0110이었고, 대응 부트스트랩 95% 신뢰구간은 [−0.0316, +0.0091]이었다.

**학습 횟수.** 기본 다언어 LSTM을 같은 20,125쌍으로 처음부터 학습하면서 최대 에포크를 50에서 100으로 늘렸다. 영–한 Macro-F1은 **0.6164에서 0.6553**으로 올랐다. 차이는 +0.0389였고, 95% 신뢰구간은 **[+0.0121, +0.0656]**이었다. 세 조합 평균은 +0.0095 올랐고, 신뢰구간 [−0.0046, +0.0238]은 0을 포함했다. 이 구간은 시드 42로 학습한 모델의 테스트 표본 변동을 반영한다.

**학습 데이터의 가짓수.** 언어 조합과 레이블별로 추출한 10,063쌍을 100회 학습한 실행과, 전체 20,125쌍을 50회 학습한 실행은 각각 가중치를 500번 업데이트했다. 검증에서 고른 모델의 세 조합 평균 테스트 점수는 각각 0.6789와 0.7056이었다. 학습 표본에 따라 단어장도 달라진다. 이 결과를 바탕으로 **서로 다른 주제와 표현을 담은 영–한 질문쌍을 추가하면 새로운 질문에 대한 성능도 향상될 수 있다**는 가설을 세웠다. 새로운 질문쌍 수집 및 독립 평가셋 검증은 후속 실험이다.

## 코드와 데이터

- [원본 구현](original/)과 [영어 기준 모델 A](original_fixed/)는 영어 Word2Vec, 공유 LSTM, 맨해튼 유사도를 사용한다.
- [다언어 모델 B](improved/)는 정렬 fastText 벡터와 공유 LSTM을 사용하며 마지막 유효 은닉 상태로 질문을 표현한다. [모델 C](improved_attention/)는 질문 내부의 모든 유효 시점에 가중치를 매겨 문장 벡터를 구한다.
- [질문쌍 데이터](dataset/)에는 영–영, 한–한, 영–한 분할이 있다. 영–한 자료는 별도로 추출한 QQP 영어 질문쌍의 둘째 질문을 한국어로 번역해 만들었다. 중복 제거 후 학습 20,125쌍, 검증 2,081쌍, 테스트 2,156쌍이다.
- [언어별 지표와 혼동행렬](results/test_scores.csv), [실행별 검증·테스트 점수](results/run_summary.csv), [질문쌍별 예측과 대응 비교](results/)를 제공한다. 절반 실험의 학습 행은 [분할 목록](improved/split_manifest_half.json)에서 확인할 수 있다.

## 결과 재계산

Python과 NumPy가 있으면 다음 명령으로 저장된 예측에서 지표를 재계산할 수 있다. 비교 코드는 같은 질문을 공유하는 쌍을 한 묶음으로 처리하고 각 언어 조합 안에서 대응 표본을 10,000회 추출한다.

```sh
python results/compare.py
```

## 학습과 평가

실험 환경은 Windows, Python 3.10, TensorFlow 2.10.1, CUDA 11.2, cuDNN 8.1이다. 시드는 42, 배치 크기는 2,048이며 Adam과 MSE를 사용했다. 기본 다언어 LSTM의 50에포크 모델과 100에포크 모델에서 선택한 임계값은 각각 0.61과 0.58이다. 학습된 가중치는 로컬의 각 모델 실행 폴더에 저장된다.

```powershell
conda env create -f original_fixed/environment.yml
conda activate malstm-gpu
python -m nltk.downloader stopwords

python original_fixed/train.py
python original_fixed/evaluate.py
python improved/train.py
python improved/evaluate.py
python improved_attention/train.py
python improved_attention/evaluate.py

python improved/train.py --epochs 100 --run-dir improved/artifacts/full_100
python improved/evaluate.py --run-dir improved/artifacts/full_100

python improved/train.py --epochs 100 --split-manifest improved/split_manifest_half.json --run-dir improved/artifacts/half_100
python improved/evaluate.py --split-manifest improved/split_manifest_half.json --run-dir improved/artifacts/half_100
```

학습 시 영어 기준 모델은 `embedding/GoogleNews-vectors-negative300.bin.gz`를 사용한다. 다언어 모델은 `embedding/wiki.en.align.vec`와 `embedding/wiki.ko.align.vec`를 사용한다. 학습 후에는 가중치와 어휘표만으로 예측할 수 있다.

```powershell
python improved/predict.py --run-dir improved/artifacts/full_100 --languages ko en --questions "비밀번호를 어떻게 바꾸나요?" "How do I change my password?"
```

## 출처

- 참조 코드 · [likejazz/Siamese-LSTM](https://github.com/likejazz/Siamese-LSTM)
- 한국어 질문쌍 · [songys/Question_pair](https://github.com/songys/Question_pair)
- 영어 질문쌍 · [Quora Question Pairs, GLUE](https://huggingface.co/datasets/nyu-mll/glue/viewer/qqp)
- 한국어·영어 임베딩 · [fastText 정렬 벡터](https://fasttext.cc/docs/en/aligned-vectors.html)
- 영어 임베딩 · [Google word2vec](https://code.google.com/archive/p/word2vec/) · [Google News 벡터 파일](https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz)
