import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from app.config import settings
from app.models.schemas import DocumentProcessingJob


class StructuredExporter:
    """Exports processed document results into JSON, CSV, and Excel formats."""

    def __init__(self, export_dir: Path = settings.EXPORTS_DIR):
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, job: DocumentProcessingJob) -> Dict[str, str]:
        """
        Exports job results to JSON, CSV, and Excel files.
        Returns dictionary of filepaths: {'json': path, 'csv': path, 'excel': path}.
        """
        base_filename = f"job_{job.job_id}"
        
        # 1. JSON Export
        json_path = self.export_dir / f"{base_filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(job.model_dump_json(indent=2))

        # Flatten field results for tabular formats
        tabular_data = []
        for field in job.extracted_fields:
            tabular_data.append({
                "Job ID": job.job_id,
                "Template ID": job.template_id,
                "Field ID": field.field_id,
                "Label": field.label,
                "Field Type": field.field_type.value,
                "Raw Text": field.raw_text,
                "Normalized Text": field.normalized_text,
                "OCR Confidence": field.ocr_confidence,
                "Validation Passed": field.validation_passed,
                "Final Confidence": field.final_confidence,
                "Human Review Flag": field.human_review_flag,
                "Choice Group": field.choice_group_id,
                "Choice Option": field.choice_option_value,
                "Choice Selected": field.choice_selected,
                "Choice Group Status": field.choice_group_status,
            })

        df = pd.DataFrame(tabular_data)

        # 2. CSV Export
        csv_path = self.export_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig ensures proper Burmese script rendering in Excel/CSV

        # 3. Excel Export
        excel_path = self.export_dir / f"{base_filename}.xlsx"
        df.to_excel(excel_path, index=False, engine="openpyxl")

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "excel": str(excel_path),
        }
