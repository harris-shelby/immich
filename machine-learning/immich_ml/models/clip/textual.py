import requests

import json
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding, Tokenizer

from immich_ml.config import log
from immich_ml.models.base import InferenceModel
from immich_ml.models.constants import WEBLATE_TO_FLORES200
from immich_ml.models.transforms import clean_text, serialize_np_array
from immich_ml.schemas import ModelSession, ModelTask, ModelType


class BaseCLIPTextualEncoder(InferenceModel):
    depends = []
    identity = (ModelType.TEXTUAL, ModelTask.SEARCH)
    DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"  # DeepL API 地址

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepl_api_key = "fe07bcee-c6a7-5fcc-4c0d-3edd06fc8165:fx"  # 配置 DeepL API 密钥

    def _predict(self, inputs: str, language: str | None = None, **kwargs: Any) -> str:
         # 检测中文并翻译
        if language == "zh" and self.contains_chinese(inputs):
            log.debug("Translating Chinese input to English...")
            inputs = self.translate_to_english(inputs)
        tokens = self.tokenize(inputs, language=language)
        res: NDArray[np.float32] = self.session.run(None, tokens)[0][0]
        return serialize_np_array(res)
   
  def contains_chinese(self, text: str) -> bool:
        """检查文本中是否包含中文字符"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def translate_to_english(self, text: str) -> str:
        """使用 DeepL API 将文本翻译为英语"""
        try:
            response = requests.post(
                self.DEEPL_API_URL,
                data={
                    "auth_key": self.deepl_api_key,
                    "text": text,
                    "source_lang": "ZH",
                    "target_lang": "EN-US",
                },
            )
            response.raise_for_status()
            return response.json()["translations"][0]["text"]
        except Exception as e:
            log.warning(f"Translation failed: {e}")
            return text  # 如果翻译失败，返回原文本
  
    def _load(self) -> ModelSession:
        session = super()._load()
        log.debug(f"Loading tokenizer for CLIP model '{self.model_name}'")
        self.tokenizer = self._load_tokenizer()
        tokenizer_kwargs: dict[str, Any] | None = self.text_cfg.get("tokenizer_kwargs")
        self.canonicalize = tokenizer_kwargs is not None and tokenizer_kwargs.get("clean") == "canonicalize"
        self.is_nllb = self.model_name.startswith("nllb")
        log.debug(f"Loaded tokenizer for CLIP model '{self.model_name}'")

        return session

    @abstractmethod
    def _load_tokenizer(self) -> Tokenizer:
        pass

    @abstractmethod
    def tokenize(self, text: str, language: str | None = None) -> dict[str, NDArray[np.int32]]:
        pass

    @property
    def model_cfg_path(self) -> Path:
        return self.cache_dir / "config.json"

    @property
    def tokenizer_file_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    @property
    def tokenizer_cfg_path(self) -> Path:
        return self.model_dir / "tokenizer_config.json"

    @cached_property
    def model_cfg(self) -> dict[str, Any]:
        log.debug(f"Loading model config for CLIP model '{self.model_name}'")
        model_cfg: dict[str, Any] = json.load(self.model_cfg_path.open())
        log.debug(f"Loaded model config for CLIP model '{self.model_name}'")
        return model_cfg

    @property
    def text_cfg(self) -> dict[str, Any]:
        text_cfg: dict[str, Any] = self.model_cfg["text_cfg"]
        return text_cfg

    @cached_property
    def tokenizer_file(self) -> dict[str, Any]:
        log.debug(f"Loading tokenizer file for CLIP model '{self.model_name}'")
        tokenizer_file: dict[str, Any] = json.load(self.tokenizer_file_path.open())
        log.debug(f"Loaded tokenizer file for CLIP model '{self.model_name}'")
        return tokenizer_file

    @cached_property
    def tokenizer_cfg(self) -> dict[str, Any]:
        log.debug(f"Loading tokenizer config for CLIP model '{self.model_name}'")
        tokenizer_cfg: dict[str, Any] = json.load(self.tokenizer_cfg_path.open())
        log.debug(f"Loaded tokenizer config for CLIP model '{self.model_name}'")
        return tokenizer_cfg


class OpenClipTextualEncoder(BaseCLIPTextualEncoder):
    def _load_tokenizer(self) -> Tokenizer:
        context_length: int = self.text_cfg.get("context_length", 77)
        pad_token: str = self.tokenizer_cfg["pad_token"]

        tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_file_path.as_posix())

        pad_id: int = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(length=context_length, pad_token=pad_token, pad_id=pad_id)
        tokenizer.enable_truncation(max_length=context_length)

        return tokenizer

    def tokenize(self, text: str, language: str | None = None) -> dict[str, NDArray[np.int32]]:
        text = clean_text(text, canonicalize=self.canonicalize)
        if self.is_nllb and language is not None:
            flores_code = WEBLATE_TO_FLORES200.get(language)
            if flores_code is None:
                no_country = language.split("-")[0]
                flores_code = WEBLATE_TO_FLORES200.get(no_country)
                if flores_code is None:
                    log.warning(f"Language '{language}' not found, defaulting to 'en'")
                    flores_code = "eng_Latn"
            text = f"{flores_code}{text}"
        tokens: Encoding = self.tokenizer.encode(text)
        return {"text": np.array([tokens.ids], dtype=np.int32)}


class MClipTextualEncoder(OpenClipTextualEncoder):
    def tokenize(self, text: str, language: str | None = None) -> dict[str, NDArray[np.int32]]:
        text = clean_text(text, canonicalize=self.canonicalize)
        tokens: Encoding = self.tokenizer.encode(text)
        return {
            "input_ids": np.array([tokens.ids], dtype=np.int32),
            "attention_mask": np.array([tokens.attention_mask], dtype=np.int32),
        }
