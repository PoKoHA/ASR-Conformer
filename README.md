Paper:https://arxiv.org/abs/2005.08100

## Model
### - Encoder
![0](https://user-images.githubusercontent.com/76771847/126898639-5ade49ba-39cf-457d-8ec7-710f7a700096.png)

**Conv Module**
![c](https://user-images.githubusercontent.com/76771847/126898664-ae760179-4577-44d1-83b1-82e080c79d9d.png)

**Attention Module**
![f](https://user-images.githubusercontent.com/76771847/126898677-dd02db3f-6d80-4cc8-9d8c-5d46ce1d9571.png)

**FFN Module**
![e](https://user-images.githubusercontent.com/76771847/126898690-9369d95c-9d6f-462c-8831-59fe2a323065.png)

## Update

Conformer 에서는 Decoder로 LSTM만 사용

=> Decoder 에 ASR-Transformer Decoder 추가


## Reference

lite transformer with long-short range attention
: https://arxiv.org/abs/2004.11886

specAugment
:https://arxiv.org/abs/1904.08779

Multi-head attention with relative pos encoding
:https://arxiv.org/abs/1901.02860

https://github.com/lucidrains/conformer

https://github.com/sooftware/conformer

## Dataset

clovacall: https://github.com/clovaai/ClovaCall

Ai_hub(1,000 hours)








