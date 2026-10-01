from transformers import PretrainedConfig


class BorealisConfig(PretrainedConfig):
    model_type = "borealis"

    def __init__(
        self,
        whisper_encoder_name: str = "openai/whisper-large-v3",
        llm_name: str = "unsloth/Qwen2.5-0.5B-Instruct",
        downsample_factor: int = 4,
        **kwargs,
    ):
        self.whisper_encoder_name = whisper_encoder_name
        self.llm_name = llm_name
        self.downsample_factor = downsample_factor
        super().__init__(**kwargs)


BorealisConfig.register_for_auto_class()
