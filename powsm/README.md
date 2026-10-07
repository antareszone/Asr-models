---
datasets:
- anyspeech/ipapack_plus_train_1
- anyspeech/ipapack_plus_train_2
- anyspeech/ipapack_plus_train_3
- anyspeech/ipapack_plus_train_4
language: multilingual
library_name: espnet
license: cc-by-4.0
metrics:
  - pfer
  - cer
tags:
  - espnet
  - audio
  - phone-recognition
  - automatic-speech-recognition
  - grapheme-to-phoneme
  - phoneme-to-grapheme
pipeline_tag: automatic-speech-recognition
---

### 🐁POWSM

<p align="left">
  <a href="https://arxiv.org/abs/2510.24992"><img src="https://img.shields.io/badge/Paper-2510.24992-red.svg?logo=arxiv&logoColor=red"/></a>
  <a href="https://huggingface.co/espnet/powsm"><img src="https://img.shields.io/badge/Model-powsm-yellow.svg?logo=huggingface&logoColor=yellow"/></a>
  <a href="https://github.com/espnet/egs2/powsm/s2t1"><img src="https://img.shields.io/badge/Recipe-powsm-blue.svg?logo=github&logoColor=black"/></a>
</p>

POWSM is the first phonetic foundation model that can perform four phone-related tasks:
Phone Recognition (PR), Automatic Speech Recognition (ASR), audio-guided grapheme-to-phoneme conversion (G2P), and audio-guided phoneme-to-grapheme
conversion (P2G).

Based on [Open Whisper-style Speech Model (OWSM)](https://www.wavlab.org/activities/2024/owsm/) and trained with [IPAPack++](https://huggingface.co/anyspeech), POWSM outperforms or matches specialized PR models of similar size while jointly supporting G2P, P2G, and ASR.

> [!TIP]
> Check out our new model: [🐁POWSM-CTC](https://huggingface.co/espnet/powsm_ctc), an encoder-only variant based on OWSM-CTC structure,
> and [💎PRiSM](https://arxiv.org/abs/2601.14046): Benchmarking Phone Realization in Speech Models! 


To use the pre-trained model, please install `espnet` and `espnet_model_zoo`. The requirements are:
```
torch
espnet
espnet_model_zoo
```

**The recipe can be found in ESPnet:** https://github.com/espnet/espnet/tree/master/egs2/powsm/s2t1

### Example script for PR/ASR/G2P/P2G

Our models are trained on 16kHz audio with a fixed duration of 20s. When using the pre-trained model, please ensure the input speech is 16kHz and pad or truncate it to 20s.

To distinguish phone entries from BPE tokens that share the same Unicode, we enclose every phone in slashes and treat them as special tokens. For example, /pʰɔsəm/ would be tokenized as /pʰ//ɔ//s//ə//m/. 


> [!NOTE]
> Jan 2026: We release a retrained version with improved ASR text normalization. 
> It is located in the subfolder `textnorm_retrained` and has the same structure as the main model.
> Additional details are provided in the updated arXiv appendix.

```python
from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf  # or librosa

task = "<pr>"
s2t = Speech2Text.from_pretrained(
    "espnet/powsm",
    device="cuda",
    lang_sym="<eng>",   # ISO 639-3; set to <unk> for unseen languages
    task_sym=task,    # <pr>, <asr>, <g2p>, <p2g>
)

speech, rate = sf.read("sample.wav")
prompt = "<na>"         # G2P: set to ASR transcript; P2G: set to phone transcription with slashes
pred = s2t(speech, text_prev=prompt)[0][0]

# post-processing for better format
pred = pred.split("<notimestamps>")[1].strip()
if task == "<pr>" or task == "<g2p>":
  pred = pred.replace("/", "")
print(pred)
```

#### Other tasks

See `force_align.py` in [ESPnet recipe](https://github.com/espnet/espnet/tree/master/egs2/powsm/s2t1) to try out CTC forced alignment with POWSM's encoder!

LID is learned implicitly during training, and you may run it with the script below:

```python
from espnet2.bin.s2t_inference_language import Speech2Language
import soundfile as sf      # or librosa

s2t = Speech2Language.from_pretrained(
    "espnet/powsm",
    device="cuda",
    nbest=1,                # number of possible languages to return
    first_lang_sym="<afr>", # fixed; defined in vocab list
    last_lang_sym="<zul>"   # fixed; defined in vocab list
)

speech, rate = sf.read("sample.wav")
pred = model(speech)[0]     # a list of lang-prob pair
print(pred)
```

### Citations

```BibTex
@article{powsm,
      title={POWSM: A Phonetic Open Whisper-Style Speech Foundation Model}, 
      author={Chin-Jou Li and Kalvin Chang and Shikhar Bharadwaj and Eunjung Yeo and Kwanghee Choi and Jian Zhu and David Mortensen and Shinji Watanabe},
      year={2025},
      eprint={2510.24992},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24992}, 
}
```