# 영어 중복 질문 판별 · MaLSTM 기준 모델

프로젝트 루트 `nlp`에서 아래 명령을 실행한다. 학습은 `dataset/train/english_train.txt`의 7,000쌍, 검증은 `dataset/validation/english_validation.txt`의 700쌍을 사용한다. 테스트 700쌍은 학습 후 별도 스크립트로 평가한다.

## 처음 한 번 준비하기

Windows 64비트와 NVIDIA GPU용 Anaconda 환경이다. TensorFlow가 Windows에서 CUDA를 직접 지원하는 2.10 계열을 사용한다. NVIDIA 그래픽 드라이버가 설치된 PC에서 실행한다. [TensorFlow의 Windows 설치 안내](https://www.tensorflow.org/install/pip#windows-native)

```powershell
conda env create -f original_fixed/environment.yml
conda activate malstm-gpu
python -m nltk.downloader stopwords
```

영어 Google News Word2Vec 파일(압축 약 1.65 GB)을 한 번 준비한다. 아래 명령은 기본 위치에 파일을 내려받는다. 이미 파일이 있다면 학습 명령에 `--vectors "D:\경로\GoogleNews-vectors-negative300.bin.gz"`를 추가해도 된다. [벡터 파일](https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/blob/main/GoogleNews-vectors-negative300.bin.gz)

```powershell
curl.exe --fail --location --retry 3 --continue-at - --create-dirs --output embedding/GoogleNews-vectors-negative300.bin.gz https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz
```

## GPU 학습

```powershell
python original_fixed/train.py
```

기본값은 GPU 한 장, 50 epoch, 배치 크기 2,048, 시드 42다. 시작할 때 TensorFlow가 선택한 GPU 이름을 출력한다. GPU를 찾지 못하면 환경을 확인할 수 있도록 중단한다. CPU가 필요할 때는 `--device cpu`로 지정한다. GPU 메모리가 부족하면 `--batch-size 512`로 낮추고 실험에 사용한 값을 기록한다. 임베딩 파일은 순차로 읽고 학습 어휘에 있는 벡터만 메모리에 보관한다.

학습은 매 epoch마다 검증 점수를 계산한다. 0.00부터 1.00까지 0.01 간격으로 임계값을 바꾸어 검증 Macro-F1이 가장 큰 조합을 저장한다. 같은 epoch에서 동점이면 0.50에 가장 가까운 임계값을, 거리까지 같으면 더 작은 값을 택한다. epoch 간 동점은 먼저 나온 epoch를 택한다.

기본 결과 폴더는 `original_fixed/artifacts/english_baseline/`이다. `model.weights.h5`에는 고정 임베딩과 공유 LSTM 가중치가, `run.json`에는 학습 단어표와 불용어, 모델 설정, 선택한 epoch·임계값·검증 F1이 들어간다. 다른 시드로 다시 학습할 경우 새 결과 폴더를 지정하고 평가할 때도 같은 폴더를 전달한다.

```powershell
python original_fixed/train.py --seed 43 --run-dir original_fixed/artifacts/english_seed43
python original_fixed/evaluate.py --run-dir original_fixed/artifacts/english_seed43
```

## 테스트 F1 계산

```powershell
python original_fixed/evaluate.py
```

저장된 단어표·가중치·임계값을 불러와 `dataset/test/english_test.txt`에 적용한다. **Test Macro-F1**은 중복·비중복 두 클래스의 F1 평균, **Test duplicate F1**은 중복(1) 클래스의 F1이다. `english_test_predictions.tsv`에는 원본 행 순서대로 0부터 시작하는 행 번호, 두 질문, 정답, 유사도, 예측을 기록한다. 다음 모델과 같은 행의 예측을 짝지어 비교할 수 있다. 평가에는 외부 Word2Vec 파일이나 NLTK 불용어 다운로드가 필요하지 않다.

질문을 직접 넣어 유사도와 판정을 확인할 수도 있다.

```powershell
python original_fixed/predict.py --questions "How do I reset my password?" "How can I change my password?"
```

## 원본과 실험 조건

원본의 영어 정규화·불용어 제거, 고정 Word2Vec 300차원, 최대 20단어(뒤쪽 자르기, 앞쪽 0 패딩), 은닉 크기 50의 공유 LSTM, 맨해튼 유사도 `exp(-sum(abs(h1 - h2)))`, MSE와 Adam을 사용한다. 준비된 분할을 직접 읽고 학습 자료에서만 단어 번호를 매긴다. 학습에 없는 단어는 별도 UNK 번호로 처리한다. 두 GPU를 전제한 원본 학습 코드를 단일 GPU용으로 바꾸고, 검증 Macro-F1으로 모델을 선택한다. [원본 코드 · likejazz/Siamese-LSTM](https://github.com/likejazz/Siamese-LSTM/)
