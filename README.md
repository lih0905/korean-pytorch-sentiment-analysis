# PyTorch와 TorchText를 이용한 한국어 감정 분석 연습

Ben Trevett(@bentrevett)의 [감정 분석 tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis)을 토대로 한국어 데이터셋([네이버 영화 리뷰](https://github.com/e9t/nsmc))을 이용한 감정 분석 연습 레포입니다.

한국어 토크나이저는 [KoNLPy](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)의 은전한닢(Mecab)을 사용했습니다.


> 권장 패키지(최신화 기준)
>  * torch>=2.2
>  * torchdata>=0.7
>  * konlpy>=0.6
>  * transformers>=4.38
>  * pandas>=2.0


## 최신 버전 전환 검토

최신 PyTorch/Transformers 기준으로 이 튜토리얼을 업데이트할 때 필요한 변경점은 아래 문서에 정리했습니다.

- [Torch2 마이그레이션 검토 문서](docs/TORCH2_MIGRATION_REVIEW.md)


## 현대식 데이터 파이프라인 예제

`torchtext.data.Field`/`BucketIterator` 의존을 줄이기 위해 `Dataset + DataLoader + collate_fn` 기반 예제를 추가했습니다.

- 데이터 유틸: `modern/data.py`
- 베이스라인 학습 스크립트: `modern/train_baseline.py`

실행 예시:

```bash
python -m modern.train_baseline --epochs 1 --max-batches 20
```
