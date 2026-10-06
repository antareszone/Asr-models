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

### 🐁POWSM-CTC

<p align="left">
  <a href="https://arxiv.org/abs/2601.14046"><img src="https://img.shields.io/badge/Paper-2601.14046-red.svg?logo=arxiv&logoColor=red"/></a>
  <a href="https://huggingface.co/espnet/powsm_ctc"><img src="https://img.shields.io/badge/Model-powsm_ctc-yellow.svg?logo=huggingface&logoColor=yellow"/></a>
  <a href="https://github.com/changelinglab/prism"><img src="https://img.shields.io/badge/Benchmark-PRiSM-green.svg?logo=github&logoColor=black"/></a>
  <a href="https://github.com/espnet/egs2/powsm_ctc/s2t1"><img src="https://img.shields.io/badge/Recipe-powsm_ctc-blue.svg?logo=github&logoColor=black"/></a>
</p>

POWSM-CTC is a variant of [POWSM](https://huggingface.co/espnet/powsm),  the first phonetic foundation model that can perform four phone-related tasks.
Its multi-task encoder-CTC structure is based on [OWSM-CTC](https://aclanthology.org/2024.acl-long.549/), and trained on [IPAPack++](https://huggingface.co/anyspeech), the same dataset as POWSM.

This model is proposed together with our paper [PRiSM](https://arxiv.org/abs/2601.14046), the first open-source benchmark for phone recognition systems. 
Its decoding is much faster than encoder-decoder models, with similar or enhanced PR performance on unseen domain.

> [!TIP]
> Check out POWSM-CTC's predecessor: [🐁POWSM](https://huggingface.co/espnet/powsm)

To use the pre-trained model, please install `espnet` and `espnet_model_zoo`. The requirements are:
```
torch
espnet
espnet_model_zoo
```

**The recipe can be found in ESPnet:** https://github.com/espnet/espnet/tree/master/egs2/powsm_ctc/s2t1


### Example script for PR/ASR/G2P/P2G
Our models are trained on 16kHz audio with a fixed duration of 20s. When using the pre-trained model, please ensure the input speech is 16kHz and pad or truncate it to 20s.
To distinguish phone entries from BPE tokens that share the same Unicode, we enclose every phone in slashes and treat them as special tokens. For example, /pʰɔsəm/ would be tokenized as /pʰ//ɔ//s//ə//m/. 
```python
from espnet2.bin.s2t_inference import Speech2Text

s2t = Speech2Text.from_pretrained(
    "espnet/powsm_ctc",
    device="cuda",
    use_flash_attn=True,
    lang_sym='<unk>',
    task_sym='<pr>',
)

res = [
    " ".join(text for _, _, text in s2t.decode_long(path, batch_size=16))
    for path in ["audio1.wav", "audio2.wav"]
]   # res is a list of str
```

### Citations

```BibTex
@article{prism,
      title={PRiSM: Benchmarking Phone Realization in Speech Models},
      author={Shikhar Bharadwaj and Chin-Jou Li and Yoonjae Kim and Kwanghee Choi and Eunjung Yeo and Ryan Soh-Eun Shim and Hanyu Zhou and Brendon Boldt and Karen Rosero Jacome and Kalvin Chang and Darsh Agrawal and Keer Xu and Chao-Han Huck Yang and Jian Zhu and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2601.14046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.14046}, 
}
