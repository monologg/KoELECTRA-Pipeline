from transformers import ElectraTokenizer, ElectraForTokenClassification
from ner_pipeline import NerPipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-naver-ner")
model = ElectraForTokenClassification.from_pretrained("monologg/koelectra-small-finetuned-naver-ner")

ner = NerPipeline(model=model,
                  tokenizer=tokenizer,
                  ignore_labels=[],
                  ignore_special_tokens=True)

texts = [
    "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다. 출처 : 미디어오늘 (http://www.mediatoday.co.kr)",
    "2017년 장점마을 문제가 본격적으로 이슈가 될 무렵 임 의원은 장점마을 민관협의회 위원들과 여러 차례 마을과 금강농산을 찾아갔다.",
    "2009년 7월 FC서울을 떠나 잉글랜드 프리미어리그 볼턴 원더러스로 이적한 이청용은 크리스탈 팰리스와 독일 분데스리가2 VfL 보훔을 거쳐 지난 3월 K리그로 컴백했다. 행선지는 서울이 아닌 울산이었다"
]

pprint(ner(texts))
