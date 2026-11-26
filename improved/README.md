# 한·영 다언어 MaLSTM

영어와 한국어 fastText 정렬 벡터를 하나의 고정 임베딩 표에 넣고, 가중치를 공유하는 LSTM으로 두 질문을 읽는다. 각 질문의 마지막 유효 은닉 상태 사이의 맨해튼 거리로 중복 유사도를 구한다. 이 모델은 다음 Attention 실험의 비교 기준이다.

## 학습과 테스트

`dataset/`, `original_fixed/`, `improved/`가 나란히 있는 프로젝트 루트에서 실행한다. 영어 기준 모델에 사용한 Anaconda 환경과 TSV 입력 검사, 평가 함수, GPU 설정, 맨해튼 유사도 층을 재사용한다.

```powershell
conda activate malstm-gpu
python improved/train.py
python improved/evaluate.py
```

학습은 기본적으로 GPU 한 장에서 배치 크기 2,048, 50 epoch, 시드 42로 수행한다. 시작할 때 GPU 이름을 표시한다. 환경을 새로 만들 때에는 `conda env create -f original_fixed/environment.yml`을 사용한다. GPU 메모리가 부족할 때에는 `--batch-size 512`로 조절하고 이후 비교 실험에서도 같은 값을 사용한다.

프로젝트 루트의 `embedding/`에서 `wiki.en.align.vec`와 `wiki.ko.align.vec`를 읽는다. 두 파일은 `improved_attention/`과 공유한다. 누락된 파일만 같은 위치에 내려받으며, 다른 위치에 있는 파일은 직접 지정할 수 있다.

```powershell
python improved/train.py --en-vectors "D:\vectors\wiki.en.align.vec" --ko-vectors "D:\vectors\wiki.ko.align.vec"
```

## 데이터와 입력 표현

파일에 기록된 질문의 위치와 언어 조합을 사용한다. 예를 들어 `cross_en_ko_train.txt`의 첫 번째 질문은 영어, 두 번째 질문은 한국어다. 영문 약어나 숫자만 있는 한국어 질문도 같은 규칙을 적용한다.

| 조합 | 학습 | 검증 | 테스트 |
|---|---:|---:|---:|
| 영–영 | 7,000 | 700 | 700 |
| 한–한 | 6,127 | 681 | 756 |
| 영–한 | 6,998 | 700 | 700 |
| 합계 | 20,125 | 2,081 | 2,156 |

`split_manifest.json`은 원본 파일의 해시와 제외할 행 번호를 기록한다. 질문 양끝 공백을 정리하고 언어와 질문을 함께 비교한다. 순서를 뒤집은 동일 쌍까지 확인해 테스트, 검증, 학습 순으로 첫 번째 행을 남겼다. 학습에서는 학습·검증 파일만 연다. 데이터셋을 교체했을 때는 `python improved/prepare_splits.py`로 목록을 새로 만든다. 이후 모델에서도 동일한 목록을 사용해야 행별 예측을 비교할 수 있다.

- 유니코드 정규화와 영문 소문자 변환 후 공백·문장부호를 기준으로 나눈다. 한글, 숫자, 부정 표현과 단어 안의 아포스트로피를 보존한다. 한국어 조사와 어미는 단어에 붙은 채 처리한다.
- 언어별 학습 어휘 중 해당 언어의 정렬 파일에 있는 단어에 고유 ID를 준다. 같은 철자도 영어와 한국어 ID가 다르며 두 언어의 300차원 벡터는 같은 공간에 놓인다. 파일은 순차로 읽고 필요한 벡터만 메모리에 남긴다. 학습 자료에서 실제 벡터를 찾은 토큰 수와 비율을 영어·한국어별로 출력하고 저장한다.
- 미등록 단어의 ID는 1이고 벡터는 매칭된 학습 어휘 벡터의 평균이다. 패딩 ID는 0이다. 앞 20개 토큰을 사용하고 오른쪽에 패딩을 채운 뒤 LSTM에서 마스킹한다.
- 세 언어 조합을 합쳐 epoch마다 섞어 학습한다. 공유 LSTM의 은닉 크기는 50이고 유사도는 `exp(-sum(abs(h1 - h2)))`다. MSE 손실과 Adam을 사용한다.

## 결과와 시연

각 epoch에서 하나의 공통 임계값을 0.00~1.00 사이에서 0.01 간격으로 탐색한다. 세 언어 조합별 검증 Macro-F1의 단순평균이 가장 높은 epoch와 임계값을 선택한다. 임계값 동점은 0.50에 가까운 값, 거리가 같으면 작은 값을 고른다. epoch 동점은 앞선 epoch를 택한다.

`improved/artifacts/multilingual/`에 `model.weights.h5`와 `run.json`이 생성된다. 두 파일에 고정 임베딩, 학습된 LSTM 가중치, 언어별 단어표, 임계값과 검증 결과가 들어간다. 다시 학습하려면 `--run-dir`에 새 폴더를 지정한다. 해당 실행을 평가하거나 시연할 때도 같은 경로를 전달한다.

평가 스크립트는 조합별 Macro-F1과 중복 클래스 F1, 세 조합 Macro-F1의 단순평균을 출력하고 `test_metrics.json`에 기록한다. 언어별 `*_test_predictions.tsv`에는 원본 파일에서 0부터 센 `row_id`, 두 질문, 정답, 유사도, 예측이 들어간다. 영어 700쌍의 행 번호는 `original_fixed`의 테스트 예측과 일치한다. 가중치와 단어표를 저장한 후에는 평가와 시연에 대형 벡터 파일이 필요하지 않다.

```powershell
python improved/predict.py --languages ko en --questions "비밀번호를 어떻게 바꾸나요?" "How do I change my password?"
```

## 출처

- 기반 모델 · [likejazz/Siamese-LSTM](https://github.com/likejazz/Siamese-LSTM/)
- 한·영 벡터 · [fastText Aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html), [영어](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec), [한국어](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ko.align.vec). 벡터의 배포 라이선스는 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)이다.
- 정렬 및 임베딩 방법 · [Joulin et al., 2018](https://arxiv.org/abs/1804.07745), [Bojanowski et al., 2017](https://arxiv.org/abs/1607.04606)
- 질문쌍 출처와 번역·라벨 처리 · [프로젝트 데이터 설명](../dataset/README.md)
