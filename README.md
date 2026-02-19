# PyTorch와 TorchText를 이용한 한국어 감정 분석 연습

Ben Trevett(@bentrevett)의 [감정 분석 tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis)을 토대로 한국어 데이터셋([네이버 영화 리뷰](https://github.com/e9t/nsmc))을 이용한 감정 분석 연습 레포입니다.

한국어 토크나이저는 [KoNLPy](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)의 은전한닢(Mecab)을 사용했습니다.


> 사용 패키지
>  * torch==1.3.0
>  * torchtext==0.4.0
>  * konlpy==0.5.1
>  * transformers==2.5.1
>  * pandas==0.25.2


## 최신 버전 전환 검토

최신 PyTorch/Transformers 기준으로 이 튜토리얼을 업데이트할 때 필요한 변경점은 아래 문서에 정리했습니다.

- [Torch2 마이그레이션 검토 문서](docs/TORCH2_MIGRATION_REVIEW.md)
