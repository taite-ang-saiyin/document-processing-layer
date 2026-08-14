import pytest

from app.api.endpoints import documents
from app.services.template_references import template_reference_store


@pytest.fixture(autouse=True)
def isolate_generated_artifacts(tmp_path, monkeypatch):
    """Keep test crops, exports, and reference images out of the working tree."""
    crops_dir = tmp_path / "crops"
    exports_dir = tmp_path / "exports"
    references_dir = tmp_path / "template_references"
    for directory in (crops_dir, exports_dir, references_dir):
        directory.mkdir()

    monkeypatch.setattr(documents.pipeline.crop_engine, "output_dir", crops_dir)
    monkeypatch.setattr(documents.pipeline.exporter, "export_dir", exports_dir)
    monkeypatch.setattr(documents.exporter, "export_dir", exports_dir)
    monkeypatch.setattr(template_reference_store, "storage_dir", references_dir)
