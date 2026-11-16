"""Wrapper around TrOCR (https://huggingface.co/docs/transformers/model_doc/trocr).

TrOCR recognizes a single line of text per image. For now the whole page is passed
in; cropping pages into text regions is a later configuration step.
"""

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"


class TextReader:
    def __init__(self, model_name: str = MODEL_NAME):
        try:
            # Normal run: load from the local Hugging Face cache, no network access.
            self._load(model_name, local_files_only=True)
        except OSError:
            # First run: model not cached yet, download it once.
            self._load(model_name, local_files_only=False)

    def _load(self, model_name: str, local_files_only: bool) -> None:
        self.processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=local_files_only)

    def read(self, image: Image.Image) -> str:
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
