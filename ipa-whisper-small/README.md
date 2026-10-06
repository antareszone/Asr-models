---
language:
- af
- am
- ar
- as
- az
- ba
- be
- bg
- bn
- ca
- cs
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- gl
- gn
- ha
- he
- hi
- ht
- hu
- hy
- ia
- id
- is
- it
- ja
- kk
- ko
- ky
- lo
- ltg
- lt
- lv
- mk
- ml
- mn
- mr
- mt
- nan
- nl
- pa
- pl
- ps
- pt
- ro
- ru
- sd
- sk
- sl
- sq
- sr
- sv
- sw
- ta
- te
- th
- tk
- tr
- tt
- ug
- uk
- ur
- uz
- vi
- yo
- yue
- zh
- zu
- tn
tags:
- audio
- automatic-speech-recognition
- IPA
- phonetic
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
model-index:
- name: ipa-whisper-small
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech (clean)
      type: librispeech_asr
      config: clean
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 99999999999999
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech (other)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 99999999999999
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice 11.0
      type: mozilla-foundation/common_voice_11_0
      config: hi
      split: test
      args:
        language: hi
    metrics:
    - name: Test WER
      type: wer
      value: 99999999999999
pipeline_tag: automatic-speech-recognition
license: apache-2.0
base_model:
- openai/whisper-small
---

# Whisper IPA

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Fine-tuned on 15000 wavs 
of labelled synthetic IPA data (generated using the goruut 0.6.3 phonemizer), Whisper models demonstrate a strong ability
to generalise to many languages, datasets and domains **without** the need for fine-tuning.

Whisper was proposed in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) 
by Alec Radford et al from OpenAI. The original code repository can be found [here](https://github.com/openai/whisper).

**Disclaimer**: Content for this model card has partly been written by the Hugging Face team, and parts of it were 
copied and pasted from the original model card.

## Fine-tuning details

- Fine-tuning took 66:07:22
- It was trained on 15000 wavs
- GPU in use was NVIDIA 3090ti with 24GB VRAM
- Fine-tuned on 15000 random wavs from common voice 21 across 70+ languages

## Model details

Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model. 
It was trained on 680k hours of labelled speech data annotated using large-scale weak supervision. 

The models were trained on either English-only data or multilingual data. The English-only models were trained 
on the task of speech recognition. The multilingual models were trained on both speech recognition and speech 
translation. For speech recognition, the model predicts transcriptions in the *same* language as the audio. 
For speech translation, the model predicts transcriptions to a *different* language to the audio.

Whisper checkpoints come in five configurations of varying model sizes.
The smallest four are trained on either English-only or multilingual data.
The largest checkpoints are multilingual only. All ten of the pre-trained checkpoints 
are available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The 
checkpoints are summarised in the following table with links to the models on the Hub:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |

# Usage

To transcribe audio samples, the model has to be used alongside a [`WhisperProcessor`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor).

The `WhisperProcessor` is used to:
1. Pre-process the audio inputs (converting them to log-Mel spectrograms for the model)
2. Post-process the model outputs (converting them from tokens to text)

The model is informed of which task to perform (transcription or translation) by passing the appropriate "context tokens". These context tokens 
are a sequence of tokens that are given to the decoder at the start of the decoding process, and take the following order:
1. The transcription always starts with the `<|startoftranscript|>` token
2. The second token is the language token (e.g. `<|en|>` for English)
3. The third token is the "task token". It can take one of two values: `<|transcribe|>` for speech recognition or `<|translate|>` for speech translation
4. In addition, a `<|notimestamps|>` token is added if the model should not include timestamp prediction

Thus, a typical sequence of context tokens might look as follows:
```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
```
Which tells the model to decode in English, under the task of speech recognition, and not to predict timestamps.

These tokens can either be forced or un-forced. If they are forced, the model is made to predict each token at 
each position. This allows one to control the output language and task for the Whisper model. If they are un-forced, 
the Whisper model will automatically predict the output langauge and task itself.

The context tokens can be set accordingly:

```python
model.config.forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="english", task="transcribe")
```

Which forces the model to predict in English under the task of speech recognition.

## Transcription

### Speech to IPA 
In this example, the context tokens are 'unforced', meaning the model automatically predicts the output language
(English) and task (transcribe).

```python
>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> # load model and processor
>>> processor = WhisperProcessor.from_pretrained("neurlang/ipa-whisper-small")
>>> model = WhisperForConditionalGeneration.from_pretrained("neurlang/ipa-whisper-small")
>>> model.config.forced_decoder_ids = None
>>> model.config.suppress_tokens = []
>>> model.generation_config.forced_decoder_ids = None
>>> model.generation_config._from_model_config = True

>>> # load dummy dataset and read audio files
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> sample = ds[0]["audio"]
>>> input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

>>> # generate token ids
>>> predicted_ids = model.generate(input_features)
>>> # decode token ids to text
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
['mˈɪstɚ kwˈɪltɚ ˈɪz ðə ˈeɪ pˈɑsəl ˈʌv ðə ˈmɪdəl klˈæsɪz ˈænd wˈɪɹ glæd tˈu ˈælkəm ˈhɪz gˈʌsbəl']

>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
['mˈɪstɚ kwˈɪltɚ ˈɪz ðə ˈeɪ pˈɑsəl ˈʌv ðə ˈmɪdəl klˈæsɪz ˈænd wˈɪɹ glæd tˈu ˈælkəm ˈhɪz gˈʌsbəl']
```
The context tokens can be removed from the start of the transcription by setting `skip_special_tokens=True`.



## Long-Form Transcription

The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking 
algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers 
[`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) 
method. Chunking is enabled by setting `chunk_length_s=30` when instantiating the pipeline. With chunking enabled, the pipeline 
can be run with batched inference. It can also be extended to predict word level timestamps by passing `return_timestamps="word"`:

```python
>>> import torch
>>> from transformers import pipeline
>>> from datasets import load_dataset

>>> device = "cuda:0" if torch.cuda.is_available() else "cpu"

>>> pipe = pipeline(
>>>   "automatic-speech-recognition",
>>>   model="neurlang/ipa-whisper-small",
>>>   chunk_length_s=30,
>>>   device=device,
>>> )

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> sample = ds[0]["audio"]

>>> prediction = pipe(sample.copy(), batch_size=8)["text"]
"mˈɪstɚ kwˈɪltɚ ˈɪz ðɪ əpˈɑsəl əv ðə ˈmɪdəl klˈæsɪz ˈænd wˈɪɹ glˈæd tˈɪ wˈɛlkəm ˈhɪz gˈɑspəl"

>>> # we can also return timestamps for the predictions
>>> prediction = pipe(sample.copy(), batch_size=8, return_timestamps="word")["chunks"]

[{'text': 'mˈɪstɚ', 'timestamp': (0.42, 0.78)}, {'text': ' kwˈɪltɚ', 'timestamp': (0.78, 1.2)}, {'text': ' ˈɪz', 'timestamp': (1.2, 1.4)}, {'text': ' ðɪ', 'timestamp': (1.4, 1.52)}, {'text': ' əpˈɑsəl', 'timestamp': (1.52, 2.08)}, {'text': ' əv', 'timestamp': (2.08, 2.26)}, {'text': ' ðə', 'timestamp': (2.26, 2.36)}, {'text': ' ˈmɪdəl', 'timestamp': (2.36, 2.6)}, {'text': ' klˈæsɪz', 'timestamp': (2.6, 3.22)}, {'text': ' ˈænd', 'timestamp': (3.22, 3.42)}, {'text': ' wˈɪɹ', 'timestamp': (3.42, 3.66)}, {'text': ' glˈæd', 'timestamp': (3.66, 4.02)}, {'text': ' tˈɪ', 'timestamp': (4.02, 4.18)}, {'text': ' wˈɛlkəm', 'timestamp': (4.18, 4.58)}, {'text': ' ˈhɪz', 'timestamp': (4.58, 4.82)}, {'text': ' gˈɑspəl', 'timestamp': (4.82, 5.38)}]
```

Refer to the blog post [ASR Chunking](https://huggingface.co/blog/asr-chunking) for more details on the chunking algorithm.

## Fine-Tuning

The pre-trained Whisper model demonstrates a strong ability to generalise to different datasets and domains. However, 
its predictive capabilities can be improved further for certain languages and tasks through *fine-tuning*. The blog 
post [Fine-Tune Whisper with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) provides a step-by-step 
guide to fine-tuning the Whisper model with as little as 5 hours of labelled data.

### Evaluated Use

The primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.

The models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.

In particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes.


## Training Data

The models are trained on 680,000 hours of audio and the corresponding transcripts collected from the internet. 65% of this data (or 438,000 hours) represents English-language audio and matched English transcripts, roughly 18% (or 126,000 hours) represents non-English audio and English transcripts, while the final 17% (or 117,000 hours) represents non-English audio and the corresponding transcript. This non-English data represents 98 different languages. 

As discussed in [the accompanying paper](https://cdn.openai.com/papers/whisper.pdf), we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.


## Performance and Limitations

Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. 

However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.

Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in [the paper accompanying this release](https://cdn.openai.com/papers/whisper.pdf). 

In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in [the paper](https://cdn.openai.com/papers/whisper.pdf). It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages.


## Broader Implications

We anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications.

There are also potential dual use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects.


### BibTeX entry and citation info
```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```