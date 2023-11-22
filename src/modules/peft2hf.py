from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class HuggingFaceHugs(LLM):
  pipeline: Any
  class Config:
    """Configuration for this pydantic object."""
    extra = Extra.forbid

  def __init__(self, model, tokenizer, task="text-generation"):
    super().__init__()
    self.pipeline = pipeline(task, model=model, tokenizer=tokenizer)

  @property
  def _llm_type(self) -> str:
    """Return type of llm."""
    return "huggingface_hub"

  def _call(self, prompt, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,):
    # Runt the inference.
    text = self.pipeline(prompt, max_length=100)[0]['generated_text']
    
    # @alvas: I've totally no idea what this in langchain does, so I copied it verbatim.
    if stop is not None:
      # This is a bit hacky, but I can't figure out a better way to enforce
      # stop tokens when making calls to huggingface_hub.
      text = enforce_stop_tokens(text, stop)
    print(text)
    return text[len(prompt):]