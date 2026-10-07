---
license: cc0-1.0
language:
- en
- de
- fr
- es
- ru
- it
pipeline_tag: automatic-speech-recognition
tags:
- Phone Recognition
- International Phonetic Alphabet
- CTC
- multilingual
datasets:
- pklumpp/CommonPhoneDataset
base_model:
- facebook/wav2vec2-large-xlsr-53
---
# Model Card for Wav2Vec2 Large with Common Phone

This is a multilingual phone recognition model optimized with the [Common Phone](https://zenodo.org/records/5846137) dataset.
It was created in the scope of the PhD thesis [Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech](https://open.fau.de/items/d0c6b800-e217-4049-ab1f-a746fc9b3966) by [Philipp Klumpp](https://scholar.google.com/citations?user=IWvgno4AAAAJ) to analyze pathological speech signals.

Find the Source Code to use this model on [**GITHUB**](https://github.com/PKlumpp/phd_model).

To cite this work, please use the following BibTex snippet:

```
@phdthesis{klumpp2024phdthesis,
  author  = "Philipp Klumpp",
  title   = "Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech",
  school  = "Friedrich-Alexander-Universit{\"a}t Erlangen-N{\"u}rnberg",
  address = "Erlangen, Germany",
  year    = 2024,
  month   = may
}
```


## Model Details

Wav2Vec2 model with linear projection to CTC blank token + 101 phone symbols from the International Phonetic Alphabet (IPA).
The model uses 16 kHz audio to predict the most probable sequence of uttered IPA phones.

### Model Description

This model was created to analyze pathological speech signals. It was optimized with [Common Phone](https://huggingface.co/datasets/pklumpp/CommonPhoneDataset), a multilingual corpus for robust acoustic modelling. It comprises more than 11.000 speakers which were carefully selected from Mozilla's Common Voice dataset.
Results in terms of phone error rate (PER) in percent:

| Language | Test PER |
|:---:|:---:|
| English | 11.0 |
| French | 9.9 |
| German | 9.8 |
| Italian | 9.1 |
| Russian | 6.6 |
| Spanish | 8.8 |
| **Average** | **9.2** |

- **Developed by:** [Philipp Klumpp](https://scholar.google.com/citations?user=IWvgno4AAAAJ)
- **Model type:** [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- **Languages:** Multilingual (English, French, German, Italian, Russian, Spanish)
- **License:** [Creative Commons Zero 1.0 (CC0)](https://creativecommons.org/publicdomain/zero/1.0/deed.en)
- **Finetuned from model:** [Wav2Vec2 XLSR-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
- **Finetuning dataset:** [Common Phone](https://huggingface.co/datasets/pklumpp/CommonPhoneDataset) as published in [**Common Phone: A Multilingual Dataset for Robust Acoustic Modelling**](https://arxiv.org/abs/2201.05912)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [GitHub](https://github.com/PKlumpp/phd_model)
- **Paper:** The final print of the thesis will be linked here.

## Contact

[Philipp Klumpp](mailto:philipp-klumpp@live.de)