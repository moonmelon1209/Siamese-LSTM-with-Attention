# 한·영 MaLSTM + Attention (모델 C)

`improved/`에서 학습한 모델 B와 같은 언어별 단어표 구성, 정렬 fastText 임베딩, 공유 LSTM을 사용한다. 각 질문을 대표하는 벡터를 만드는 방식이 실험 변수다. B는 마지막 유효 은닉 상태를 사용하고, C는 각 단어 위치의 은닉 상태에 가중치를 매겨 합한다.

## Attention의 위치

`Embedding → LSTM(return_sequences=True) → AttentionPooling → 질문 벡터`

1. 두 질문이 같은 인코더를 통과한다. 각 질문에서 LSTM은 단어 위치별 50차원 은닉 상태 `h_t`를 반환한다.
2. 각 위치의 점수는 `vᵀ tanh(W h_t + b)`다. 실제 토큰 위치의 점수에 softmax를 적용해 가중치 합을 1로 만든다. 패딩 ID 0의 가중치는 0이며 미등록 단어 ID 1은 유효 토큰으로 참여한다.
3. `sum(alpha_t * h_t)`로 질문 하나의 50차원 벡터를 만든다. 두 질문은 동일한 LSTM과 Attention 가중치를 사용하며 최종 유사도는 기존과 같은 `exp(-sum(abs(left - right)))`다. 한 질문 내부의 토큰 중요도를 학습하는 구조다.

## GPU 학습과 평가

기존 `nlp` 프로젝트 루트에서 실행한다. 같은 위치에 `original_fixed/`, `improved/`, `dataset/`이 있어야 한다.

```powershell
conda activate malstm-gpu
python improved_attention/train.py
python improved_attention/evaluate.py
```

프로젝트 루트의 `embedding/wiki.en.align.vec`와 `embedding/wiki.ko.align.vec`를 모델 B와 함께 사용한다. 학습은 사전학습된 고정 임베딩을 사용하고 LSTM·Attention은 처음부터 학습한다. 재실행 결과를 다른 폴더에 저장할 때는 학습·평가·예측 명령에 같은 `--run-dir`를 전달한다.

| 항목 | B와 C의 공통 설정 |
|---|---|
| 데이터 | 같은 분할 목록, 학습 20,125쌍 · 검증 2,081쌍 · 테스트 2,156쌍 |
| 인코더 | 300차원 고정 임베딩, 최대 20토큰, 오른쪽 패딩 마스크, 은닉 크기 50 |
| 학습 | 기본 시드 42, 배치 크기 2,048, 50 epoch, Adam, MSE |
| 모델 선택 | 세 언어 조합의 검증 Macro-F1 단순평균이 가장 높은 epoch와 공통 임계값 |
| 최종 평가 | 고정된 임계값으로 영–영, 한–한, 영–한의 테스트 Macro-F1과 중복 클래스 F1 산출 |

## 결과 확인과 시연

`improved_attention/artifacts/multilingual_attention/`에 학습이 `model.weights.h5`와 `run.json`을 저장한다. 평가는 `test_metrics.json`과 세 언어 조합의 `*_test_predictions.tsv`를 만든다.

실험의 주 비교값은 **C의 세 조합 평균 Macro-F1 − B의 세 조합 평균 Macro-F1**이다. 두 실행의 `test_metrics.json`에서 `mean_macro_f1`을 확인한다. 언어별 변화는 `by_group`에서 확인하고, 대응 비교는 양쪽 TSV에서 언어 조합과 동일한 `row_id`를 기준으로 예측을 연결한다. 가중치는 각 모델이 자신의 검증 데이터에서 선택한 체크포인트를 사용한다.

```powershell
python improved_attention/predict.py --languages ko en --questions "비밀번호를 어떻게 바꾸나요?" "How do I change my password?"
```

기반 구현은 [likejazz/Siamese-LSTM](https://github.com/likejazz/Siamese-LSTM/)이며 정렬 임베딩의 출처와 데이터 전처리 규칙은 [다언어 모델 설명](../improved/README.md)에 기록했다.
