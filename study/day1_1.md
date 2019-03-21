# 1 일차
- [ ] 품사 판별, 형태소 분석, 그리고 미등록단어 문제에 대한 글 입니다. ([링크](https://lovit.github.io/nlp/2018/04/01/pos_and_oov/))
- [ ] Java 로 구현된 Komoran 을 Jupyter notebook 의 python 에서 이용하기 위한 과정입니다. (링크)
- [ ] Java 로 구현된 Komoran 을 Python package 로 만드는 과정과 Java, Python 간의 변수 호환에 대한 내용입니다 (링크)
- [ ] 텍스트 데이터를 KoNLPy 를 이용하여 term frequency matrix 로 만드는 과정입니다. (링크)
- [ ] Logistic regression 의 이론설명 입니다. (링크)


## 1. Part of speech tagging, Tokenization, and Out of vocabulary problem

### 1.1 Tokenization, Part of speech tagging, Morphological analysis?
- 토크나이징(tokenization) : 주어진 문장을 토큰(tokens)으로 나누는 과정
```
sent = '너무너무너무는 아이오아이의 노래입니다'
tokens = ['너무너무너무는', '아이오아이의', '노래입니다']

sent = 'Very-very-very is song of I.O.I'
tokens = ['Very-very-very', 'is', 'song', 'of', 'I.O.I']
```
- 품사 판별(POS tagging: part of speech tagging) : 토큰을 (단어, 품사)로 정의
```
tokens = [
    ('너무너무너무', '명사'),
    ('는', '조사'),
    ('아이오아이', '명사'),
    ('의', '조사'),
    ('노래', '명사'),
    ('입니다', '형용사')
]
```
- 형태소 분석(morphological analysis) : 품사 판별과 자주 혼동되는 개념, 형태소란 의미를 지니는 최소 단위로, (1) 자립형태소 / 의존형태소 로 나뉘기도 하며, (2) 실질형태소와 형식형태소로 나뉘기도 합니다.

### 1.2 Out of vocabulary problem
형태소 분석이나 품사 판별을 위해서는 사전과 문법이 필요합니다. ‘아이오아이는’ 이라는 어절이 ‘아이오아이는 = 아이오아이/명사 + 는/조사’ 라는 규칙을 알고 있지 않더라도 다음과 같은 사전과 문법 규칙을 지니고 있다면, ‘아이오아이는’ 이라는 어절을 인식할 수 있습니다.
```
dictionary = {
	'명사': {'아이오아이', '아이', '오'},
	'조사': {'는'}
}

pattern = [('명사', '조사'), ('명사',)]
```
```
from konlpy.tag import Twitter

twitter = Twitter()
twitter.pos('너무너무너무는 아이오아이의 노래입니다')

[('너무', 'Noun'),
 ('너무', 'Noun'),
 ('너무', 'Noun'),
 ('는', 'Josa'),
 ('아이오', 'Noun'),
 ('아이', 'Noun'),
 ('의', 'Josa'),
 ('노래', 'Noun'),
 ('입니', 'Adjective'),
 ('다', 'Eomi')]
 ```
일단 두 가지 큰 문제가 보입니다. ‘너무너무너무’가 ‘너무 + 너무 + 너무’로 나뉘어졌습니다. ‘아이오아이’는 ‘아이오 + 아이’로 나뉘어졌습니다. 트위터분석기에 ‘너무’, ‘아이오’, ‘아이’는 명사 사전에 등록이 되었지만, ‘너무너무너무’, ‘아이오아이’는 명사 사전에 존재하지 않기 때문입니다. 
또한 미등록단어 문제들은 주로 하나의 단어가 여러 개의 잘못된 단어로 나뉘어지는 형태로 발생합니다. 형태소 분석기, 품사 판별기는 학습 데이터를 바탕으로 주어진 문장을 이해합니다. 위와 같은 결과가 나온 이유는 학습 데이터를 이용한 모델에서 score(‘너무너무너무/unknown’ + ‘는/Josa’) < score(‘너무/Noun + 너무/Noun + 너무/Noun + 는/Josa’) 이기 때문입니다. 즉 학습데이터에서 보지 못했던 단어를 모르는 단어로 인식하는 것보다 아는 단어의 조합으로 나누는 것을 더 선호하기 떄문입니다.

```
Hannanum: 한나눔 형태소 분석기
Kkma: 꼬꼬마 형태소 분석기 (Kind Korean Moorpheme Analyzer)
Komoran: 코모란 한국어 형태소 분석기
Mecab: mecab-ko 형태소 분석기
Twitter: 트위터 한국어 처리기 -> 처리기!!
```
형태소 분석기의 목표는 단어를 형태소로 분해하는 것입니다. 그리고 형태소 분석을 통하여 품사 판별을 할 때의 가정 중 하나는 이전에 없던 형태소의 결합으로 신조어도 만들 수 있다는 점입니다. 예를 들어 ‘신/관형사’, ‘메뉴/명사’ 를 사전에 알고 있고, ‘관형사 + 명사 -> 명사’라는 규칙을 안다면 ‘신메뉴’를 명사로 인식할 수 있습니다.
```
사전: {신/관형사, 메뉴/명사}
형태소규칙: '관형사 + 명사 -> 명사'

pos('신메뉴') = morpheme(신/관형사 + 메뉴/명사) = '명사'
```

### 1.3 Ambiguity in tokenization
미등록단어 문제와 비슷한 문제가 또 있습니다. 누가 했던 말인지는 모르지만, 텍스트 데이터를 다루는 사람이라며 누구라도 할 수 있는 말입니다. ‘텍스트분석, 자연어처리는 처음부터 끝까지 모호성과의 싸움‘입니다.
```
{
	'명사': {'서울', '서울대', '공원', '대공원'},
	'조사': {'에서'}
}
```
‘서울대/명사 + 공원/명사 + 에서/조사’도 가능하며, ‘서울/명사 + 대공원/명사 + 에서/조사’도 가능합니다.
형태소 분석기, 품사 판별기는 학습데이터를 바탕으로 가능한 여러 개우 후보 중에서 가장 가능성이 높은 후보를 최종 결과로 선택합니다. 만약 학습된 모델의 P(서울대 + 공원) > P(서울 + 대공원) 이었다면, 해당 어절은 ‘서울대 + 공원 + 에서’로 나뉘어질 것입니다. 설령 해당 문서가 어린이날 행사와 관련된 문서였더라도요.

### 1.4 Needs for unsupervised word extractions
모든 품사에서 새로운 단어가 만들어지는 것은 아니라는 것입니다. 일단 명사는 새로운 단어가 자주 만들어짐을 압니다. 이는 새로운 개념이 만들어지기 때문입니다. 새로운 사건, 새로운 사람을 지칭하기 위하여 명사가 만들어집니다.

하지만 조사는 새롭게 만들어지지 않습니다. 조사는 문법 기능을 담당하기 때문입니다. 문법은 규칙이며, 규칙이 바뀔 때에는 이를 이용하는 사람들 간의 합의가 필요합니다. 그렇기 때문에 조사와 같은 문법이 변하는데는 긴 시간이 필요합니다.

새로운 어미도 만들어집니다. 말투 때문입니다. ‘저녁 이제 먹을라궁’ 처럼 대화체에서는 ‘-을라궁’ 처럼 다양한 어미가 만들어지기도 합니다. 일단 어떤 품사에서 새로운 단어가 만들어지는지를 파악해야 그 방법을 설계할 수 있습니다.

### Related posts
- [Left-side subword tokenizer](https://lovit.github.io/nlp/2018/04/02/simplest_tokenizers/) 는 문서 판별 등의 작업에 이용할 수 있는 아주 간단한 토크나이저 입니다.
- [Word piece model](https://lovit.github.io/nlp/2018/04/02/wpm/) 은 out of vocabulary 문제를 우회 (해결이 아닙니다) 하는 토크나이저 입니다.
- [Cohesion score](https://lovit.github.io/nlp/2018/04/09/cohesion_ltokenizer/) 는 단어의 일부분으로 다른 부분이 얼마나 잘 예상되느냐에 대한 정보를 단어 추출에 이용합니다.
- [KR-WordRank](https://lovit.github.io/nlp/2018/04/16/krwordrank/) 는 graph ranking 방법을 이용하여 단어를 추출합니다.
- [Branching Entropy](https://lovit.github.io/nlp/2018/04/09/branching_entropy_accessor_variety/) 와 [Acessor Variety](https://lovit.github.io/nlp/2018/04/09/branching_entropy_accessor_variety/) 는 손나은의 오른쪽에 등장하는 글자의 다양성의 정보를 이용합니다.

