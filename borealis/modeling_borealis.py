import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, PreTrainedModel, WhisperFeatureExtractor
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import PyTorchModelHubMixin
from .configuration_borealis import BorealisConfig
from huggingface_hub import hf_hub_download
import os


class AudioLanguageAdapter(nn.Module):
    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class BorealisForConditionalGeneration(PreTrainedModel, PyTorchModelHubMixin):
    config_class = BorealisConfig

    def __init__(self, config: BorealisConfig, language_model=None, tokenizer=None):
        super().__init__(config)
        assert tokenizer is not None, "Tokenizer надо передать в модельку"
        self.encoder: WhisperModel = WhisperModel.from_pretrained(
            config.whisper_encoder_name
        ).encoder
        self.encoder.to(torch.bfloat16)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.llm = language_model
        self.tokenizer = tokenizer
        self.llm.resize_token_embeddings(len(tokenizer))
        self.downsample_factor = config.downsample_factor
        self.adapter = AudioLanguageAdapter(
            hidden_size=self.encoder.config.d_model * self.downsample_factor,
            dim=self.llm.config.hidden_size,
        )
        self.adapter.to(torch.bfloat16)
        self.bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

    def _downsample(self, seq: torch.Tensor) -> torch.Tensor:
        k, (T, d) = self.downsample_factor, seq.shape
        target = k * math.ceil(T / k)
        if target != T:
            seq = F.pad(seq, (0, 0, 0, target - T))
        return seq.contiguous().view(target // k, d * k)

    def _tok_embed(self, tok_id: int, batch: int, device) -> torch.Tensor:
        idx = torch.full((batch, 1), tok_id, dtype=torch.long, device=device)
        return self.llm.get_input_embeddings()(idx)

    def forward(
        self,
        mel: torch.Tensor,
        audio_att_mask: torch.Tensor,
        labels: torch.Tensor,
        text_att_mask: torch.Tensor,
    ):
        B, device = mel.size(0), mel.device
        enc_out = self.encoder(
            input_features=mel, attention_mask=None, return_dict=True
        ).last_hidden_state
        audio_embs, audio_mask, max_T = [], [], 0
        for seq in enc_out:
            ds = self._downsample(seq)
            audio_embs.append(ds)
            max_T = max(max_T, ds.size(0))
        for ds in audio_embs:
            pad = max_T - ds.size(0)
            audio_mask.append(
                torch.cat(
                    [
                        torch.ones(ds.size(0), dtype=torch.long, device=device),
                        torch.zeros(pad, dtype=torch.long, device=device),
                    ]
                )
            )
            if pad:
                ds = F.pad(ds, (0, 0, 0, pad))
        audio_embeddings = torch.stack(audio_embs, 0)
        audio_mask = torch.stack(audio_mask, 0)
        audio_embeddings = self.adapter(audio_embeddings)
        text_embeddings = self.llm.get_input_embeddings()(labels)
        sa_positions = (labels == self.audio_start_id).nonzero(as_tuple=True)
        ea_positions = (labels == self.audio_end_id).nonzero(as_tuple=True)
        inputs_embeds = []
        att_mask = []
        for b in range(B):
            sa_idx = sa_positions[1][sa_positions[0] == b].item()
            ea_idx = ea_positions[1][ea_positions[0] == b].item()
            prefix_emb = text_embeddings[b, : sa_idx + 1]
            postfix_emb = text_embeddings[b, ea_idx:]
            emb = torch.cat([prefix_emb, audio_embeddings[b], postfix_emb], dim=0)
            prefix_mask = text_att_mask[b, : sa_idx + 1]
            postfix_mask = text_att_mask[b, ea_idx:]
            full_mask = torch.cat([prefix_mask, audio_mask[b], postfix_mask], dim=0)
            inputs_embeds.append(emb)
            att_mask.append(full_mask)
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0.0
        )
        att_mask = torch.nn.utils.rnn.pad_sequence(
            att_mask, batch_first=True, padding_value=0
        )
        assistant_prompt = self.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        ).input_ids
        assistant_starts = []
        for b in range(B):
            seq = labels[b]
            for i in range(len(seq) - len(assistant_prompt)):
                if torch.equal(
                    seq[i : i + len(assistant_prompt)],
                    torch.tensor(assistant_prompt, device=device),
                ):
                    assistant_start = i + len(assistant_prompt)
                    break
            else:
                raise ValueError("Assistant prompt not found")
            assistant_starts.append(assistant_start + (ea_idx - sa_idx - 1) + max_T)
        max_len = inputs_embeds.size(1)
        loss_labels = labels.new_full((B, max_len), -100)
        for b in range(B):
            orig_assist_start = assistant_starts[b] - max_T - (ea_idx - sa_idx - 1)
            content_len = len(labels[b]) - orig_assist_start
            loss_labels[b, assistant_starts[b] : assistant_starts[b] + content_len] = (
                labels[b, orig_assist_start:]
            )
        if self.tokenizer.pad_token_id is not None:
            loss_labels[loss_labels == self.tokenizer.pad_token_id] = -100
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            labels=loss_labels,
            return_dict=True,
        )
        return out.loss, out.logits

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        att_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        return_tokens = kwargs.pop("return_tokens", False)
        single = mel.dim() == 2
        if single:
            mel, att_mask = mel.unsqueeze(0), att_mask.unsqueeze(0)
        mel = mel.to(torch.bfloat16)
        B, device = mel.size(0), mel.device
        enc_out = self.encoder(
            input_features=mel, attention_mask=None, return_dict=True
        ).last_hidden_state
        audio_embs, audio_mask, max_T = [], [], 0
        for seq in enc_out:
            ds = self._downsample(seq)
            audio_embs.append(ds)
            max_T = max(max_T, ds.size(0))
        for i, ds in enumerate(audio_embs):
            pad = max_T - ds.size(0)
            audio_mask.append(
                torch.cat(
                    [
                        torch.ones(ds.size(0), dtype=torch.long, device=device),
                        torch.zeros(pad, dtype=torch.long, device=device),
                    ]
                )
            )
            if pad:
                audio_embs[i] = F.pad(ds, (0, 0, 0, pad))
        audio_embeddings = torch.stack(audio_embs, 0)
        audio_mask = torch.stack(audio_mask, 0)
        audio_embeddings = self.adapter(audio_embeddings)
        messages = [
            {
                "role": "system",
                "content": "Вы полезный помощник по автоматическому распознаванию речи. Точно транскрибируйте аудио в текст.",
            },
            {
                "role": "user",
                "content": "Транскрибируйте это аудио: <|start_of_audio|><|end_of_audio|>",
            },
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids.repeat(B, 1)
        text_att_mask = model_inputs.attention_mask.repeat(B, 1)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        sa_idx = (input_ids[0] == self.audio_start_id).nonzero(as_tuple=True)[0].item()
        ea_idx = (input_ids[0] == self.audio_end_id).nonzero(as_tuple=True)[0].item()
        inputs_embeds = []
        full_att_mask = []
        for b in range(B):
            prefix_emb = text_embeddings[b, : sa_idx + 1]
            postfix_emb = text_embeddings[b, ea_idx:]
            emb = torch.cat([prefix_emb, audio_embeddings[b], postfix_emb], dim=0)
            prefix_mask = text_att_mask[b, : sa_idx + 1]
            postfix_mask = text_att_mask[b, ea_idx:]
            mask = torch.cat([prefix_mask, audio_mask[b], postfix_mask], dim=0)
            inputs_embeds.append(emb)
            full_att_mask.append(mask)
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0.0
        )
        att_mask = torch.nn.utils.rnn.pad_sequence(
            full_att_mask, batch_first=True, padding_value=0
        )
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        if return_tokens:
            return gen_ids[0] if single else gen_ids
        else:
            txt = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            if single:
                return txt[0]
            else:
                return [t for t in txt]

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_directory)
        extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_encoder_name
        )
        extractor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = BorealisConfig.from_pretrained(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        language_model = AutoModelForCausalLM.from_pretrained(config.llm_name)
        model = cls(config, language_model=language_model, tokenizer=tokenizer)

        state_dict_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="pytorch_model.bin"
        )

        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model


BorealisForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
