# Burmese Insurance Claim Form OCR System

Welcome to the **Burmese Insurance Claim Form OCR System** documentation repository.

## Documentation Index

- 📘 [System Architecture Specification](file:///c:/Users/Msi%20GF66/Desktop/ocr/system_architecture.md): Complete breakdown of the Template Registration, Document Processing (Inference), and Continuous Improvement pipelines.

## Architecture At A Glance

The system automates the processing of multilingual (Burmese & English) insurance claim forms through a two-stage architecture:

1. **Template Registration Subsystem**: Uses `PP-DocLayoutV3` for layout boundary coordinate ownership, multilingual OCR for printed labels, VLM for semantic mapping, and human approval to publish template definitions to the **Template Registry**.
2. **Document Processing Subsystem**: Ingests completed forms, performs quality checks, matches templates, aligns images, crops fields, and routes ROIs through specialized OCR models (Printed, Handwriting, Checkbox, Table, Signature). Standardized output is exported after passing validation and confidence checks.
3. **Continuous Improvement Loop**: Human corrections are captured into a **Correction Dataset** to retrain extraction models offline for continuous accuracy improvement.

For full architectural details and Mermaid diagrams, see [`system_architecture.md`](file:///c:/Users/Msi%20GF66/Desktop/ocr/system_architecture.md).
