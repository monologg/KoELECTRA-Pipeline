from transformers import ElectraTokenizer, ElectraForQuestionAnswering, pipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")

qa = pipeline("question-answering", tokenizer=tokenizer, model=model)

pprint(qa({
    "question": "한국의 대통령은 누구인가?",
    "context": "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다.",
}))
