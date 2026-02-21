# Torch 1.3/TorchText 0.4 → 최신 PyTorch 스택 마이그레이션 검토

이 문서는 현재 레포(노트북 중심 튜토리얼)를 **PyTorch 2.x + 최신 데이터 파이프라인**으로 옮기기 위해 필요한 변경점을 정리한 체크리스트입니다.

## 1) 현재 상태 요약

- `requirements.txt`가 `torch==1.3.0`, `torchtext==0.4.0`, `transformers==2.5.1` 기준으로 고정되어 있습니다.
- 대부분의 노트북이 구버전 `torchtext.data` API(`Field`, `LabelField`, `BucketIterator`)에 강하게 의존합니다.
- 일부 노트북은 현재 버전에서 동작하지 않는 코드/오타를 포함합니다.
  - `torch.backends.cudnn.determinisitic = True` (`deterministic` 오타)
  - `spacy.load('en')` (최신 spaCy 모델 이름 체계와 불일치)

## 2) 우선순위별 수정 포인트

### A. 의존성(가장 먼저)

권장: 레거시 API를 억지로 유지하지 말고, 데이터 로더를 현대식으로 재작성합니다.

예시 버전(권장 출발점):

- `torch>=2.2`
- `torchtext` 제거(또는 최소화)
- `torchdata>=0.7` (선택)
- `transformers>=4.38`
- `pandas>=2.0`
- `konlpy` 최신(환경별 설치 확인 필요)

> 참고: `torchtext`는 과거 `legacy` 경로가 단계적으로 제거되어 기존 `Field` 계열 코드가 최신 버전에서 그대로 동작하지 않습니다.

### B. 데이터 파이프라인 전면 교체

현재 노트북들의 핵심 병목은 아래 패턴입니다.

- `TEXT = data.Field(...)`
- `LABEL = data.LabelField(...)`
- `data.TabularDataset.splits(...)`
- `data.BucketIterator.splits(...)`

변경 권장안:

1. `pandas`/`csv` 기반으로 파일 로딩
2. 토크나이즈 함수(`mecab.morphs`) 직접 적용
3. `vocab`(딕셔너리) 직접 구축
4. `torch.utils.data.Dataset` + `DataLoader` + `collate_fn`으로 패딩/길이 처리

이렇게 바꾸면 torchtext 버전 이슈를 피하면서 코드 수명을 늘릴 수 있습니다.

### C. 모델/학습 코드 보정

- `torch.load(...)`는 장치 안전성을 위해 `map_location=device`를 명시하는 형태로 통일 권장
- 재현성 설정 오타 수정:
  - `torch.backends.cudnn.deterministic = True`
- 이진 분류의 경우:
  - 출력/로스 일관성 (`BCEWithLogitsLoss` + sigmoid 후처리) 유지 검토
- 멀티클래스의 경우:
  - `CrossEntropyLoss` 입력 shape/label dtype(`long`) 점검

### D. Transformers 튜토리얼(6번) 별도 갱신

`transformers==2.5.1` 기반 코드는 최신 4.x에서 아래 변경이 필요합니다.

- `AutoTokenizer`, `AutoModel` 계열 사용
- 반환값(dict-like) 접근 방식으로 수정 (`outputs.last_hidden_state` 등)
- 배치 인코딩을 `tokenizer(..., padding=True, truncation=True, return_tensors='pt')`로 통일

## 3) 파일별 리팩터링 난이도(대략)

- `0 - 데이터셋 csv 변환.ipynb`: 하 (torchtext 예시 셀만 교체)
- `1 - 간단한 감정 분석.ipynb`: 중 (기본 Dataset/DataLoader 도입)
- `2 - 향상된 감정 분석.ipynb`: 중상 (길이 정보, packed sequence)
- `3 - 빠른 감정 분석.ipynb`: 중 (bigram 전처리 + 임베딩)
- `4 - CNN 이용한 감정 분석.ipynb`: 중
- `5 - 다중 감성 분석.ipynb`: 중
- `6 - 트랜스포머를 이용한 감정 분석.ipynb`: 상 (transformers API 갱신량 큼)

## 4) 권장 마이그레이션 순서

1. 공통 유틸 분리: 토크나이저/어휘집/패딩 collate
2. 노트북 1→2→3→4→5 순으로 torchtext 제거
3. 마지막으로 6번(Transformers) 업그레이드
4. 각 단계마다 `requirements`와 실행 확인 셀을 동기화

## 5) 샘플 코드 스케치 (torchtext 대체)

```python
class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, vocab):
        self.texts = df["document"].fillna("").tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)


def collate_batch(batch, pad_id=1):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
    ys = torch.tensor(ys, dtype=torch.float)
    return xs, lengths, ys
```

## 6) 이번 브랜치에서 한 일

- 레포를 훑어보고 최신화 시 깨지는 지점을 문서화했습니다.
- `modern/data.py`, `modern/train_baseline.py`로 torchtext 비의존 데이터/학습 파이프라인 초안을 추가했습니다.
