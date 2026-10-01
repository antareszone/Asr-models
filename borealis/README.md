---
license: apache-2.0
language:
- ru
pipeline_tag: automatic-speech-recognition
library_name: transformers
---

[![Демо в Colab](https://img.shields.io/badge/Демо-Colab-yellow?logo=googlecolab&style=for-the-badge)](https://colab.research.google.com/drive/1ac7apyGO24iAYMwg3DLcqLZRjo-w4QWf?usp=sharing)

## Borealis 

### Описание

**Borealis** - это наша первая audio llm c ASR для русского языка. В этом репозитории представлен чекпоинт, который видел примерно `7000` часов аудио на русском языке. Важным отличием от других моделей является то, что тут есть поддержка пунктуации в распознанных аудио. Архитектура во многом вдохновлена [Voxtral](https://mistral.ai/news/voxtral), но отличается в некоторых моментах.





### Использование 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoFeatureExtractor
import torch
import librosa

model = AutoModelForCausalLM.from_pretrained("Vikhrmodels/Borealis", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Borealis")
extractor = AutoFeatureExtractor.from_pretrained("Vikhrmodels/Borealis")

generation_params = {
        "max_new_tokens": 350,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.2,
 }

model.eval()
model.to("cuda")

waveform, sr = librosa.load("path/to/your/audio.wav", sr=16_000)

proc = extractor(
        waveform,
        sampling_rate=sr,
        padding="max_length",
        max_length=480_000,
        return_attention_mask=True,
        return_tensors="pt",
    )

mel = proc.input_features.squeeze(0).to("cuda")
att_mask = proc.attention_mask.squeeze(0).to("cuda")

with torch.inference_mode():
    transcript = model.generate(mel=mel, att_mask=att_mask, **generation_params)

print(transcript)
```

### Метрики модели 

Ниже представлены замеры `Borealis` на фоне остальных открытых моделей, который поддерживают русский язык. Бенчмарк мы скоро выложим в открытый доступ

| Модель | Средний WER | RuLS | CV 22.0 | Books | Speak | Sova |
|---|---:|---:|---:|---:|---:|---:|
| **Borealis** | 6.33% | 6.39% | **2.67%** | **5.28%** | **1.95%** | 15.37% |
| GigaAM-ASR-V2-RNNT | **5.85%** | **5.24%** | 2.85% | 8.06% | 3.08% | **10.01%** |
| openai/whisper-large-v3 | 10.74% | 11.62% | 7.51% | 12.19% | 2.74% | 19.65% |
| bond005/whisper-podlodka-turbo | 9.38% | 11.91% | 6.36% | 8.96% | 3.14% | 16.55% |
| openai/whisper-large-v3-turbo | 11.30% | 11.88% | 8.17% | 13.29% | 2.80% | 20.37% |
| bond005/whisper-large-v3-ru-podlodka | 10.76% | 10.24% | 7.80% | 10.31% | 3.23% | 22.21% |
| nvidia/canary-1b-v2 | 13.52% | 20.16% | 9.12% | 11.45% | 4.97% | 21.89% |
| VOSK-model-ru-0.42 | 11.30% | 12.06% | 11.87% | 10.80% | 2.61% | 19.15% |
| GigaAM-ASR-V2-CTC | 6.45% | 5.26% | 3.42% | 7.72% | 3.01% | 12.86% |


### Авторы 
- Илья Кулешов, Vikhr Team
- Александр Николич, Vikhr Team