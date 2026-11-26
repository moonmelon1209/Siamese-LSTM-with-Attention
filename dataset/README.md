# MaLSTM 한국어·영어·한영 질문쌍 데이터셋

## 라벨 규칙

- `is_duplicate = 1`: 두 질문의 의미가 같음
- `is_duplicate = 0`: 두 질문의 의미가 다름
- 한국어 원본의 라벨은 반전하여 QQP와 동일한 규칙으로 통일했습니다.

## 데이터 구성

| 데이터 | 행 수 | 라벨 0 | 라벨 1 | 중복 질문 비율 |
|---|---:|---:|---:|---:|
| korean_train | 6,136 | 1,967 | 4,169 | 67.94% |
| english_train | 7,000 | 2,244 | 4,756 | 67.94% |
| cross_en_ko_train | 7,000 | 2,244 | 4,756 | 67.94% |
| train_all | 20,136 | 6,455 | 13,681 | 67.94% |
| korean_validation | 682 | 250 | 432 | 63.34% |
| english_validation | 700 | 257 | 443 | 63.29% |
| cross_en_ko_validation | 700 | 257 | 443 | 63.29% |
| korean_test | 758 | 250 | 508 | 67.02% |
| english_test | 700 | 231 | 469 | 67.00% |
| cross_en_ko_test | 700 | 231 | 469 | 67.00% |

위 표는 파일에 저장된 원시 행 수입니다. `train/train_all.txt`는 세 언어 조합 20,136쌍을
섞어 놓은 참고용 합본입니다. 실제 다언어 모델은 조합별 파일과
[`split_manifest.json`](../improved/split_manifest.json)을 읽어 중복을 제외한
20,125쌍으로 학습합니다. 최종 검증은 2,081쌍, 테스트는 2,156쌍입니다.

검증은 `validation` 폴더의 세 파일을 조합별로 평가합니다. 세 조합의 Macro-F1을
동일 가중치로 평균내어 체크포인트와 공통 판정 임계값을 선택합니다.

테스트는 `test` 폴더의 세 파일을 따로 평가합니다. 테스트 데이터는 모델 가중치를 학습하거나
epoch를 선택하는 데 사용하지 않고, 모든 설정을 확정한 뒤 최종 성능을 한 번 측정할 때 사용합니다.

## QQP 표본 추출

- 출처: Hugging Face `nyu-mll/glue`, subset `qqp`
- 고정 시드: `20261002`
- 영어와 한영 표본은 서로 다른 원본 QQP 행을 사용합니다.
- train, validation, test에 사용된 QQP 원본 질문쌍은 서로 겹치지 않습니다.
- 언어가 라벨을 암시하지 않도록 영어·한영 표본의 라벨 비율을 같은 분할의 한국어 데이터와 맞췄습니다.
- Hugging Face 공식 `test`는 라벨이 `-1`로 비공개이므로 사용하지 않았습니다.
- 영어 validation/test는 라벨이 공개된 공식 `validation` 40,430개에서 서로 겹치지 않게 분리했습니다.

## 한영 데이터 생성

- `question1`: 영어 원문 유지
- `question2`: 한국어로 번역
- 번역 모델: Argos Translate `translate-en_ko` 1.1
- 한국어 문자를 포함한 번역: 8,398/8,400
- 한국어 문자가 없는 번역은 숫자·약어·고유명사 중심 문장일 수 있으므로
  `metadata/cross_translation_review.tsv`에서 검토할 수 있습니다.

## 파일 형식

모델 입력 파일은 UTF-8 탭 구분 형식이며 모두 다음 세 열을 사용합니다.

```text
question1	question2	is_duplicate
```

`metadata/qqp_source_indices.tsv`에는 재현을 위한 QQP 원본 `idx`가 저장되어 있습니다.

## 원본 다운로드

- 한국어 질문쌍 · [songys/Question_pair](https://github.com/songys/Question_pair) · [원본 MIT 라이선스](https://github.com/songys/Question_pair/blob/master/LICENSE)

- Train: https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/train-00000-of-00001.parquet?download=true
- Validation: https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet?download=true

웹에서는 Hugging Face 페이지의 `Files and versions` → `qqp` 폴더에서 각 Parquet 파일의
다운로드 버튼을 누르면 됩니다.
