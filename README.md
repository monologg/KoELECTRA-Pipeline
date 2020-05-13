# KoELECTRA-Pipeline

Transformers Pipeline with KoELECTRA

## Available Pipeline

| Subtask   | Model           | Link                                                                                                       |
| --------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| NSMC      | koelectra-base  | [koelectra-base-finetuned-nsmc](https://huggingface.co/monologg/koelectra-base-finetuned-nsmc)             |
|           | koelectra-small | [koelectra-small-finetuned-nsmc](https://huggingface.co/monologg/koelectra-small-finetuned-nsmc)           |
| Naver-NER | koelectra-base  | [koelectra-base-finetuned-naver-ner](https://huggingface.co/monologg/koelectra-base-finetuned-naver-ner)   |
|           | koelectra-small | [koelectra-small-finetuned-naver-ner](https://huggingface.co/monologg/koelectra-small-finetuned-naver-ner) |

## Fix Note

### 1. NSMC

`transformers v2.9` 기준으로 `ElectraForSequenceClassification`이 지원이 되지 않습니다.

- `model.py`에 `ElectraForSequenceClassification`를 직접 구현하였습니다.

### 2. Naver-NER

하나의 Word가 여러 개의 Wordpiece로 쪼개지는 경우가 있는데, `NerPipeline`은 piece-level로 결과를 보여줍니다. 이는 추후에 단어 단위로 복원할 때 문제가 생기게 됩니다.

- `NerPipeline` 클래스를 `ner_pipeline.py`에 일부 수정하여 재구현하였습니다.
- `ignore_special_tokens`라는 인자를 추가하여, `[CLS]`와 `[SEP]` 토큰의 결과를 무시하게 처리할 수 있습니다.
- `ignore_labels=['O']`일 시 `O` tag를 제외하고 결과를 보여줍니다.

## Requirements

`transformers`가 `v2.9.0`에서 **tokenizer 관련 API, 버그 등이 개선**되어 해당 버전을 설치하는 것을 권장합니다.

- torch>=1.4.0
- transformers==2.9.0

## Run reference code

```bash
$ python3 test_nsmc.py
$ python3 test_naver_ner.py
```

### Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Pipelines Documentation](https://huggingface.co/transformers/main_classes/pipelines.html)
- [Issue for NER Pipeline](https://github.com/huggingface/transformers/issues/3548)
