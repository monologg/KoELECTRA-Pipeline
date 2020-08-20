# KoELECTRA-Pipeline

`Transformers Pipeline` with KoELECTRA

## Available Pipeline

| Subtask       | Model              | Link                                                                                                                 |
| ------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **NSMC**      | koelectra-base     | [koelectra-base-finetuned-nsmc](https://huggingface.co/monologg/koelectra-base-finetuned-nsmc)                       |
|               | koelectra-small    | [koelectra-small-finetuned-nsmc](https://huggingface.co/monologg/koelectra-small-finetuned-nsmc)                     |
| **Naver-NER** | koelectra-base     | [koelectra-base-finetuned-naver-ner](https://huggingface.co/monologg/koelectra-base-finetuned-naver-ner)             |
|               | koelectra-small    | [koelectra-small-finetuned-naver-ner](https://huggingface.co/monologg/koelectra-small-finetuned-naver-ner)           |
| **KorQuad**   | koelectra-base-v2  | [koelectra-base-v2-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v2-finetuned-korquad)           |
|               | koelectra-small-v2 | [koelectra-small-v2-distilled-korquad-384](https://huggingface.co/monologg/koelectra-small-v2-distilled-korquad-384) |

## Customized NER Pipeline

하나의 Word가 여러 개의 Wordpiece로 쪼개지는 경우가 있는데, `NerPipeline`은 piece-level로 결과를 보여줍니다. 이는 추후에 단어 단위로 복원할 때 문제가 생기게 됩니다.

- `NerPipeline` 클래스를 `ner_pipeline.py`에 일부 수정하여 재구현하였습니다.
- `ignore_special_tokens`라는 인자를 추가하여, `[CLS]`와 `[SEP]` 토큰의 결과를 무시하게 처리할 수 있습니다.
- `ignore_labels=['O']`일 시 `O` tag를 제외하고 결과를 보여줍니다.

## Requirements

- torch>=1.4.0
- transformers==3.0.2

## Run reference code

```bash
$ python3 test_nsmc.py
$ python3 test_naver_ner.py
$ python3 test_korquad.py
```

## Example

### 1. NSMC

```python
from transformers import ElectraTokenizer, ElectraForSequenceClassification, pipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-nsmc")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-finetuned-nsmc")

nsmc = pipeline(
    "sentiment-analysis",
    tokenizer=tokenizer,
    model=model
)

print(nsmc("이 영화는 미쳤다. 넷플릭스가 일상화된 시대에 극장이 존재해야하는 이유를 증명해준다."))

# Out
[{'label': 'positive', 'score': 0.8729340434074402}]
```

### 2. Naver-NER

```python
from transformers import ElectraTokenizer, ElectraForTokenClassification
from ner_pipeline import NerPipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-naver-ner")
model = ElectraForTokenClassification.from_pretrained("monologg/koelectra-small-finetuned-naver-ner")

ner = NerPipeline(model=model,
                  tokenizer=tokenizer,
                  ignore_labels=[],
                  ignore_special_tokens=True)


pprint(ner("2009년 7월 FC서울을 떠나 잉글랜드 프리미어리그 볼턴 원더러스로 이적한 이청용은 크리스탈 팰리스와 독일 분데스리가2 VfL 보훔을 거쳐 지난 3월 K리그로 컴백했다. 행선지는 서울이 아닌 울산이었다"))

# Out
[{'entity': 'DAT-B', 'score': 0.9996234178543091, 'word': '2009년'},
 {'entity': 'DAT-I', 'score': 0.93541419506073, 'word': '7월'},
 {'entity': 'ORG-B', 'score': 0.9994615912437439, 'word': 'FC서울을'},
 {'entity': 'O', 'score': 0.999957799911499, 'word': '떠나'},
 {'entity': 'LOC-B', 'score': 0.9983285069465637, 'word': '잉글랜드'},
 {'entity': 'ORG-B', 'score': 0.9989873766899109, 'word': '프리미어리그'},
 {'entity': 'ORG-B', 'score': 0.9315412044525146, 'word': '볼턴'},
 {'entity': 'ORG-I', 'score': 0.9993480443954468, 'word': '원더러스로'},
 {'entity': 'O', 'score': 0.9999217987060547, 'word': '이적한'},
 {'entity': 'PER-B', 'score': 0.9994915127754211, 'word': '이청용은'},
 {'entity': 'ORG-B', 'score': 0.999463677406311, 'word': '크리스탈'},
 {'entity': 'ORG-I', 'score': 0.999179482460022, 'word': '팰리스와'},
 {'entity': 'LOC-B', 'score': 0.9977350234985352, 'word': '독일'},
 {'entity': 'ORG-B', 'score': 0.9813936352729797, 'word': '분데스리가2'},
 {'entity': 'ORG-B', 'score': 0.8733143210411072, 'word': 'VfL'},
 {'entity': 'ORG-I', 'score': 0.9937891960144043, 'word': '보훔을'},
 {'entity': 'O', 'score': 0.9999728202819824, 'word': '거쳐'},
 {'entity': 'DAT-B', 'score': 0.9963461756706238, 'word': '지난'},
 {'entity': 'DAT-I', 'score': 0.9909392595291138, 'word': '3월'},
 {'entity': 'ORG-B', 'score': 0.9995419383049011, 'word': 'K리그로'},
 {'entity': 'O', 'score': 0.9999108910560608, 'word': '컴백했다.'},
 {'entity': 'O', 'score': 0.9993030428886414, 'word': '행선지는'},
 {'entity': 'ORG-B', 'score': 0.9915705323219299, 'word': '서울이'},
 {'entity': 'O', 'score': 0.9999194741249084, 'word': '아닌'},
 {'entity': 'ORG-B', 'score': 0.9994401931762695, 'word': '울산이었다'}]
```

### 3. KorQuad

```python
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, pipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")

qa = pipeline("question-answering", tokenizer=tokenizer, model=model)

pprint(qa({
    "question": "한국의 대통령은 누구인가?",
    "context": "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다.",
}))

# Out
{'answer': '문재인', 'end': 3, 'score': 0.9644287549022144, 'start': 0}
```

### Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Pipelines Documentation](https://huggingface.co/transformers/main_classes/pipelines.html)
- [Issue for NER Pipeline](https://github.com/huggingface/transformers/issues/3548)
