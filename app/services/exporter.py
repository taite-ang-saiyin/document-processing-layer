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
                "Page": field.page,
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
                "Table Parent Field ID": field.table_parent_field_id,
                "Table Parent Label": field.table_parent_label,
                "Table Row Index": field.table_row_index,
                "Table Column Index": field.table_column_index,
                "Table Cell Order": field.table_cell_order,
                "Table Is Header": field.table_is_header,
                "Table Is Empty": field.table_is_empty,
                "Table Change Ratio": field.table_change_ratio,
                "Reference Difference Path": field.reference_difference_path,
            })

        df = pd.DataFrame(tabular_data)

        # 2. CSV Export
        csv_path = self.export_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig ensures proper Burmese script rendering in Excel/CSV

        # 3. Excel Export
        excel_path = self.export_dir / f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Fields", index=False)
            used_sheet_names = {"Fields"}
            for table_index, table in enumerate(job.tables, 1):
                base_name = self._safe_sheet_name(
                    table.label or table.table_parent_field_id,
                    fallback=f"Table {table_index}",
                )
                sheet_name = base_name
                suffix = 2
                while sheet_name in used_sheet_names:
                    suffix_text = f" {suffix}"
                    sheet_name = f"{base_name[:31 - len(suffix_text)]}{suffix_text}"
                    suffix += 1
                used_sheet_names.add(sheet_name)
                grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
                for cell in table.cells:
                    if cell.row_index < table.row_count and cell.column_index < table.column_count:
                        grid[cell.row_index][cell.column_index] = cell.normalized_text
                pd.DataFrame(grid).to_excel(
                    writer, sheet_name=sheet_name, index=False, header=False
                )

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "excel": str(excel_path),
        }

    @staticmethod
    def _safe_sheet_name(value: str, fallback: str) -> str:
        invalid = set('[]:*?/\\')
        cleaned = "".join("_" if character in invalid else character for character in value).strip()
        return (cleaned or fallback)[:31]
