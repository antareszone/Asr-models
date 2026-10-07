---
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- phone-recognition
- ipa
- ctc
- multilingual
- xeus
---

# PhoneticXeus

Multilingual phone recognition that turns speech into **IPA** phones, built on
the [XEUS](https://huggingface.co/espnet/xeus) speech encoder. Trained on 70+
languages (IPAPack++).

- 🎤 **Try it in your browser:** [demo Space](https://huggingface.co/spaces/changelinglab/PhoneticXeus)
- 💻 **Code & training recipe:** [GitHub](https://github.com/changelinglab/PhoneticXeus)
- 📄 **Paper:** [arXiv 2603.29042](https://arxiv.org/abs/2603.29042)

## Usage

```bash
pip install torch torchaudio transformers huggingface_hub safetensors soundfile numpy pyyaml typeguard
```

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "changelinglab/PhoneticXeus", trust_remote_code=True
).eval()

wav, sr = torchaudio.load("audio.wav")
wav = wav.mean(0)                                    # mono, shape (samples,)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

print(model.transcribe(wav, sampling_rate=16000)[0]["processed_transcript"])
# e.g. "aɪhædðætkʰjʊɹiɑsətipɪsaɪd…"
```

`model.transcribe(...)` returns a list of dicts with `processed_transcript`
(joined IPA) and `predicted_transcript` (slash-separated phones). Calling
`model(input_values)` returns frame-level CTC logits `(batch, frames, 428)`
for custom decoding.

Audio must be **mono 16 kHz**. The first load asks you to allow the repo's
remote code (`trust_remote_code=True`).

## Citation

```bibtex
@misc{pxeus26,
      title={An Empirical Recipe for Universal Phone Recognition},
      author={Shikhar Bharadwaj and Chin-Jou Li and Kwanghee Choi and Eunjung Yeo and William Chen and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2603.29042},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29042},
}
```
## License
CC-by-NC SA 4.0, please note this is an update.