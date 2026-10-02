---
license: mit
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
library_name: nemo
language:
- en
- es
- de
- fr
- it
- ar
- ja
- ko
- pt
- ru
- hi
- zh
- vi
- he
- nl
- cs
- da
- pl
- 'no'
- sv
- th
- tr
- bg
- el
- et
- fi
- hr
- hu
- lt
- lv
- ro
- sk
- uk
- mt
- sl
datasets:
- nvidia/Granary
- multilingual_librispeech
- fleurs
- mozilla-foundation/common_voice_8_0
- voxpopuli
- europarl
thumbnail: null
tags:
- speech-recognition
- cache-aware ASR
- automatic-speech-recognition
- streaming-asr
- multilingual
- speech
- audio
- FastConformer
- RNNT
- Parakeet
- ASR
- pytorch
- NeMo
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
model-index:
- name: nemotron-asr-streaming-multilingual-0.6b
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (English)
      type: google/fleurs
      config: en_us
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 7.91
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Spanish)
      type: google/fleurs
      config: es_419
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 4.11
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (French)
      type: google/fleurs
      config: fr_fr
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 9.03
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Italian)
      type: google/fleurs
      config: it_it
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 4.25
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Portuguese)
      type: google/fleurs
      config: pt_br
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 5.48
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (German)
      type: google/fleurs
      config: de_de
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 8.31
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Hindi)
      type: google/fleurs
      config: hi_in
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 6.81
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Korean)
      type: google/fleurs
      config: ko_kr
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 7.12
metrics:
- wer
pipeline_tag: automatic-speech-recognition
base_model:
- nvidia/nemotron-3.5-asr-streaming-0.6b
---

# Nemotron 3.5 ASR ONNX

<style>
h1, h2, h3, h4, h5, h6 {
  color: #76b900; /* NVIDIA green */
  font-weight: 700;
}

hr {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 2rem 0;
}

/* Improve list spacing */
ul, ol {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
}

/* Badge alignment consistency */
img {
  display: inline;
  vertical-align: middle;
}
</style>


> [!Note]
> This model is the quantized INT4 ONNX version of [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), adding language-ID prompt conditioning to support transcription across **40 language-locales** from a single model. It supports streaming inference with 0.56 seconds of latency and is simshipped alongside the baseline NVIDIA model.
> 
**Nemotron 3.5 ASR** is a multilingual, streaming Automatic Speech Recognition (ASR) model engineered to deliver high-quality multilingual transcription across both low-latency streaming and high-throughput batch workloads. Developed by NVIDIA, this 600M parameter model transcribes speech into text with native support for punctuation and capitalization, and offers runtime flexibility with configurable chunk sizes, including 80ms, 160ms, 320ms, 560ms, and 1120ms. 
This ONNX model was exported with optimization for the 560ms chunk size.

By leveraging a state-of-the-art **Cache-Aware FastConformer-RNNT** architecture, the model eliminates redundant overlapping computations common in traditional "buffered" streaming. This allows it to process only new audio chunks while reusing cached encoder context, significantly improving computational efficiency and minimizing end-to-end delay without sacrificing accuracy.

It was trained on a massive ASR dataset and is engineered to perform across diverse and challenging acoustic conditions.

This model is ready for commercial use.

---

This model is for transcription of multilingual audio.

## Release Date

- Hugging Face [06/04/2026] via https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4

## Why Choose Nemotron 3.5 ASR?

- 🌍 **Single Multilingual Model:** Transcribes 40 language-locales from one model through language-ID prompt conditioning, with optional automatic language detection.
- ⚡ **Native Streaming Architecture:** Cache-aware design enables efficient processing of continuous audio streams, designed and optimized for low-latency voice agent applications.
- 💰 **Improved Operational Efficiency:** Delivers superior throughput compared to traditional buffered streaming approaches. This allows for a higher number of parallel streams within the same GPU memory constraints, directly reducing operational costs for production environments.
- 🎛️ **Dynamic Runtime Flexibility:** Choose the optimal operating point on the latency-accuracy Pareto curve at inference time. No re-training is required to adjust for different use-case requirements.
- 📝 **Punctuation & Capitalization:** Built-in support for punctuation and capitalization in output text.

---

## Supported Languages

The model supports **40 language-locales** in total, across three tiers:

- **Transcription-ready (19 locales):** highest-accuracy ASR, ready out of the box.
- **Broad-coverage (13 locales):** production ASR across an additional 13 locales.
- **Adaptation-ready (8 locales):** recognized by the tokenizer; fine-tune on in-domain data to unlock full transcription.

| Tier | Languages (locales) |
| :--- | :--- |
| **Transcription-ready (19 locales)** | English (en-US, en-GB), Spanish (es-US, es-ES), French (fr-FR, fr-CA), Italian (it-IT), Portuguese (pt-BR, pt-PT), Dutch (nl-NL), German (de-DE), Turkish (tr-TR), Russian (ru-RU), Arabic (ar-AR), Hindi (hi-IN), Japanese (ja-JP), Korean (ko-KR), Vietnamese (vi-VN), Ukrainian (uk-UA) |
| **Broad-coverage (13 locales)** | Polish (pl-PL), Swedish (sv-SE), Czech (cs-CZ), Norwegian Bokmål (nb-NO), Danish (da-DK), Bulgarian (bg-BG), Finnish (fi-FI), Croatian (hr-HR), Slovak (sk-SK), Mandarin (zh-CN), Hungarian (hu-HU), Romanian (ro-RO), Estonian (et-EE) |
| **Adaptation-ready (8 locales)** | Greek (el-GR), Lithuanian (lt-LT), Latvian (lv-LV), Maltese (mt-MT), Slovenian (sl-SI), Hebrew (he-IL), Thai (th-TH), Norwegian Nynorsk (nn-NO) |

> **Note:** Transcription-ready and broad-coverage locales (**32 total**) produce ASR transcription out of the box; adaptation-ready locales require fine-tuning on in-domain data to enable full transcription. The model supports uppercase and lowercase letters, punctuation, spaces, and apostrophes.

## How to Use this Model

The model is available for use in via https://github.com/microsoft/onnxruntime-genai.

---

## Results (for the ONNX model)

| # | Lang | Metric | FLEURS | CV | MLS | VoxPopuli |
|---|------|--------|--------|------|------|-----------|
| 1 | ko (ko_kr) | CER | 6.19 | – | – | – |
| 2 | it (it_it) | WER | 6.69 | – | – | – |
| 3 | es (es_419) | WER | 7.28 | 9.16 | 6.07 | 9.96 |
| 4 | pt (pt_br) | WER | 9.05 | 12.08 | 8.85 | – |
| 5 | hi (hi_in) | WER | 9.79 | – | – | – |
| 6 | de (de_de) | WER | 12.25 | 12.80 | 9.91 | 16.90 |
| 7 | ar (ar_eg) | WER | 12.31 | – | – | – |
| 8 | fr (fr_fr) | WER | 12.55 | 14.38 | 8.61 | 11.99 |
| 9 | ru (ru_ru) | WER | 13.52 | – | – | – |
| 10 | vi (vi_vn) | WER | 14.45 | – | – | – |
| 11 | ja (ja_jp) | CER | 15.12 | – | – | – |
| 12 | nl (nl_nl) | WER | 16.27 | 14.29 | 18.73 | 28.89 |
| 13 | tr (tr_tr) | WER | 16.69 | – | – | – |
| 14 | uk (uk_ua) | WER | 18.16 | – | – | – |
| 15 | pl (pl_pl) | WER | 21.53 | 26.85 | 16.31 | 35.35 |
| 16 | zh (cmn_hans_cn) | CER | 22.88 | – | – | – |
| 17 | sk (sk_sk) | WER | 23.54 | – | – | – |
| 18 | nb (nb_no) | WER | 23.49 | – | – | – |
| 19 | cs (cs_cz) | WER | 25.66 | – | – | – |
| 20 | fi (fi_fi) | WER | 25.96 | – | – | – |
| 21 | bg (bg_bg) | WER | 26.35 | – | – | – |
| 22 | sv (sv_se) | WER | 28.94 | – | – | – |
| 23 | hr (hr_hr) | WER | 31.43 | – | – | 49.88 |
| 24 | da (da_dk) | WER | 33.78 | – | – | – |
| 25 | hu (hu_hu) | WER | 34.97 | – | – | – |
| 26 | ro (ro_ro) | WER | 35.58 | – | – | – |
| 27 | et (et_ee) | WER | 36.51 | – | – | – |
| 28 | el (el_gr) | WER | 48.25 | – | – | – |
| 29 | he (he_il) | WER | 51.24 | – | – | – |
| 30 | lt (lt_lt) | WER | 53.00 | – | – | – |
| 31 | sl (sl_si) | WER | 56.89 | – | – | – |
| 32 | mt (mt_mt) | WER | 62.49 | – | – | – |
| 33 | lv (lv_lv) | WER | 87.87 | – | – | – |
| 34 | th (th_th)* | CER | 100.00* | – | – | – |

## Questions and issues

For any questions open an issue or send an email to Nenad Banfic (nebanfic@microsoft.com).