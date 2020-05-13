from transformers import ElectraTokenizer, pipeline
from model import ElectraForSequenceClassification
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-nsmc")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-finetuned-nsmc")

nsmc = pipeline(
    "sentiment-analysis",
    tokenizer=tokenizer,
    model=model
)

texts = [
    "이 영화는 미쳤다. 넷플릭스가 일상화된 시대에 극장이 존재해야하는 이유를 증명해준다.",
    "촬영감독의 영혼까지 갈아넣은 마스터피스",
    "보면서 화가날수있습니다.",
    "아니 그래서 무슨말이 하고싶은거야 ㅋㅋㅋ"
]

pprint(nsmc(texts))
