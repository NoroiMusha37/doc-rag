import logging
import time
from pathlib import Path

import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from src.doc_rag.config import settings
from src.doc_rag.ingestion.exceptions import ParsingError

logger = logging.getLogger(__name__)


class DocumentParser:
    """Converts a file into a DoclingDocument. Build once; the models stay in memory."""

    def __init__(self) -> None:
        options = PdfPipelineOptions()
        options.do_table_structure = True
        options.do_formula_enrichment = settings.DOCLING_FORMULA_ENRICHMENT
        options.generate_picture_images = True
        options.images_scale = 2.0
        options.document_timeout = settings.DOCLING_TIMEOUT_SECONDS
        options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        started = time.perf_counter()
        try:
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
            # Load the layout, table and formula models now, not on the first upload.
            self._converter.initialize_pipeline(InputFormat.PDF)
        except Exception as exc:
            raise ParsingError(f"Could not load the Docling models: {exc}") from exc
        logger.info(
            f"Docling ready in {time.perf_counter() - started:.1f}s "
            f"(formula enrichment: {settings.DOCLING_FORMULA_ENRICHMENT}, "
            f"cuda: {torch.cuda.is_available()})"
        )

    def parse(self, path: Path) -> DoclingDocument:
        if not path.is_file():
            raise ParsingError(f"File not found: {path.name}")

        started = time.perf_counter()
        try:
            result = self._converter.convert(path, raises_on_error=False)
        except Exception as exc:
            raise ParsingError(f"Could not read {path.name}: {exc}") from exc
        finally:
            # Hand cached VRAM back so the embedder has room on an 8 GB card.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        errors = "; ".join(error.error_message for error in result.errors) or "no details"
        if result.status is ConversionStatus.PARTIAL_SUCCESS:
            # Timeout or some pages failed: keep what was parsed instead of losing the file.
            logger.warning(f"Parsed {path.name} only partially: {errors}")
        elif result.status is not ConversionStatus.SUCCESS:
            raise ParsingError(f"Could not convert {path.name} ({result.status.value}): {errors}")

        pages = len(result.document.pages)
        logger.info(f"Parsed {path.name}: {pages} pages in {time.perf_counter() - started:.1f}s")
        return result.document
