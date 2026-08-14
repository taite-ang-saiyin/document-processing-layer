# Burmese Insurance Claim Form OCR System

Welcome to the **Burmese Insurance Claim Form OCR System** documentation repository.

## Documentation Index

- 📘 [System Architecture Specification](file:///c:/Users/Msi%20GF66/Desktop/ocr/system_architecture.md): Complete breakdown of the Template Registration, Document Processing (Inference), and Continuous Improvement pipelines.

## Architecture At A Glance

The system automates the processing of multilingual (Burmese & English) insurance claim forms through a two-stage architecture:

1. **Template Registration Subsystem**: Uses `PP-DocLayoutV3` for layout boundary coordinate ownership, multilingual OCR for printed labels, VLM for semantic mapping, and human approval to publish template definitions to the **Template Registry**.
2. **Document Processing Subsystem**: Ingests completed forms, performs quality checks, matches templates, aligns images, crops fields, and routes ROIs through specialized OCR models (Printed, Handwriting, Checkbox, Table, Signature). Standardized output is exported after passing validation and confidence checks.
3. **Continuous Improvement Loop**: Human corrections are captured into a **Correction Dataset** to retrain extraction models offline for continuous accuracy improvement.

## Processing prerequisite: approved reference image

Each registered template must have a canonical blank-form reference image before
completed documents can be processed. The image dimensions must exactly match the
template's `width` and `height`.

```bash
curl -X POST http://localhost:8000/api/v1/templates/<template_id>/reference \
  -F "file=@approved-blank-form.png;type=image/png"
```

Processing rejects templates without a reference image (`409`) and returns a
failed job when image alignment falls below the configured threshold.

`POST /api/v1/documents/process` accepts an optional `template_id`. When it is
omitted, the service ranks all templates with registered references using ORB
feature matching and selects one only when the best score meets
`TEMPLATE_MATCH_SCORE_THRESHOLD` (default `0.50`) and leads the runner-up by
`TEMPLATE_MATCH_MARGIN` (default `0.10`). Uncertain matches return `422` with
ranked candidate scores; they are not guessed.

For full architectural details and Mermaid diagrams, see [`system_architecture.md`](file:///c:/Users/Msi%20GF66/Desktop/ocr/system_architecture.md).
