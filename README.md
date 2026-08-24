# Document processing layer

This FastAPI service executes approved extraction templates against completed insurance forms.
It does not create templates or run blank-form VLM analysis. The umbrella orchestrator performs
registration/review and sends the approved pixel-based template definition here.

In the unified Compose stack, `SEED_DEFAULT_TEMPLATE=false`. Therefore completed-document
processing uses the fields and stable `field_id` values from the newly approved template version.
The local `claim_form_v1` seed remains available only for standalone development and tests.

## Run and inspect

```bash
docker compose up --build -d
curl http://localhost:8000/health
```

Standalone Swagger UI is at `http://localhost:8000/docs`. In the umbrella stack the direct
host port is `8001`; application clients should normally use the orchestrator on port `8000`.

The Docker image installs Poppler's `pdftoppm` utility after Python dependencies. This supplies
PDF rendering without adding another Python PDF package or changing the existing PyTorch build.

## Register an approved template

`POST /api/v1/templates/register` accepts page dimensions and integer pixel bounding boxes:

```bash
curl -X POST http://localhost:8001/api/v1/templates/register \
  -H 'Content-Type: application/json' \
  -d '{
    "template_id": "vehicle_damage_claim_v1",
    "name": "Vehicle damage claim",
    "width": 1200,
    "height": 1600,
    "pages": [
      {"page_number": 1, "width": 1200, "height": 1600},
      {"page_number": 2, "width": 1200, "height": 1600}
    ],
    "fields": [
      {
        "id": "field_policy_number",
        "label": "Policy number",
        "field_type": "printed_text",
        "page": 1,
        "bbox": {"x": 680, "y": 190, "width": 360, "height": 58},
        "required": true
      },
      {
        "id": "field_contact_number",
        "label": "Contact number",
        "field_type": "handwriting",
        "page": 2,
        "bbox": {"x": 250, "y": 620, "width": 410, "height": 64},
        "required": false
      }
    ]
  }'
```

Rules enforced by the Pydantic model:

- `pages` must be sequential, starting at page 1.
- Every field has a positive one-based `page`; omitted `page` remains compatible with an old
  single-page template and defaults to 1.
- `bbox.x`, `y`, `width`, and `height` are integers in pixels, not normalized values.
- Every box must have positive area and lie inside the dimensions of its referenced page.
- Supported `field_type` values are `printed_text`, `handwriting`, `checkbox`, `table`,
  and `signature`.

The umbrella approval adapter converts the editor's normalized decimal boxes to bounded integer
pixels using each page's own width and height. Sending fractional values directly produces a 422
validation response, because an extraction crop must resolve to exact pixels.

## Process a completed multi-page PDF

Register the template first, then upload the completed form with that downstream template ID:

```bash
curl -X POST http://localhost:8001/api/v1/documents/process \
  -F 'template_id=vehicle_damage_claim_v1' \
  -F 'file=@completed_vehicle_damage_claim.pdf'
```

The endpoint returns a completed job synchronously. Each extracted field includes `page`, so
identical coordinates on separate pages remain unambiguous. Retrieve the record or exports with:

```bash
curl http://localhost:8001/api/v1/documents/jobs/JOB_ID
curl -o result.json http://localhost:8001/api/v1/documents/jobs/JOB_ID/export/json
curl -o result.csv http://localhost:8001/api/v1/documents/jobs/JOB_ID/export/csv
curl -o result.xlsx http://localhost:8001/api/v1/documents/jobs/JOB_ID/export/excel
```

An image upload supplies only page 1. A template whose fields reference later pages therefore
requires a PDF containing at least that many pages; otherwise the returned job has `FAILED`
status and explains the page-count mismatch.

## How multi-page processing is constructed

1. The API reads the upload. Images are decoded with OpenCV; PDFs are rendered to ordered PNG
   pages with `pdftoppm -png -r 144` in a temporary directory.
2. The pipeline derives the required page count from the highest `TemplateField.page`.
3. Quality checks run on every required page and are aggregated into one job-level result.
4. Canonical pages from visual-field detection are resized to their registered template dimensions;
   they are not feature-warped again. Direct raw API uploads retain the legacy ORB/homography
   alignment path.
5. For every field, the pipeline selects `aligned_pages[field.page]`, crops its integer ROI,
   retains the raw crop, cleans it, detects visual text lines for text-bearing fields, and sends
   each reliable line to OCR in reading order. If line detection is unavailable or uncertain, it
   OCRs the cleaned full crop instead. Checkboxes and signatures bypass line detection.
6. When `LLM_POST_CORRECTION_ENABLED=true`, the service sends one full aligned page plus every
   eligible field's pixel box, raw OCR text, confidence, and validation rule to the internal VLM
   adapter. A suggestion is accepted only above the configured confidence threshold and when it
   passes the field's `validation_regex`. Raw OCR is retained, and any accepted VLM change always
   requires human review.
7. Any failed quality check, validation failure, or low-confidence field sets
   `needs_human_review`; otherwise the job is `COMPLETED`.
8. JSON, UTF-8 CSV, and Excel artifacts are written to the configured export directory.

This design keeps page identity on the template field rather than encoding it into coordinates.
The same crop code can therefore operate on single- and multi-page forms.

## OCR crop artifacts

Every processed field retains its raw crop. Text-bearing fields also retain a cleaned crop and,
when detected, the individual OCR line crops. In the unified Docker deployment, inspect them on
the host under `runtime/document-crops`, `runtime/document-preprocessed-crops`, and
`runtime/document-line-crops` respectively. The job response records these paths as
`crop_image_path`, `preprocessed_crop_path`, and `line_crop_paths`.

## Runtime limitations

Templates and jobs are currently stored in process memory. A service restart clears them. The
umbrella orchestrator keeps the immutable approved definition and re-registers it before each
document request, which makes that workflow resilient to a downstream restart. Direct callers
must perform the same template and reference registration steps.

The public processing endpoint accepts an optional `template_id`. When omitted, the service
ranks templates that have registered references and selects one only when its score meets
`TEMPLATE_MATCH_SCORE_THRESHOLD` and leads the runner-up by `TEMPLATE_MATCH_MARGIN`.
Uncertain matches return a 422 response with ranked candidates.

The handwriting and table routes currently reuse TrOCR, while checkbox
and signature routes use simple pixel-density checks. Treat these as replaceable engine slots,
not production-grade final models.

## Optional Qwen3-VL post-correction

The feature is off by default. To enable it in the unified stack, configure the internal VLM
adapter with `VLM_POST_CORRECTION_PROVIDER=gemini`, `GEMINI_GOOGLE_CLOUD_PROJECT`,
`GEMINI_GOOGLE_CLOUD_LOCATION=global`, and `GOOGLE_APPLICATION_CREDENTIALS_HOST_PATH`, then set
`LLM_POST_CORRECTION_ENABLED=true`.
The adapter sends each full page plus its field OCR results to Vertex AI Gemini using Application Default Credentials (ADC). The
correction endpoint is internal and authenticated with the shared `VLM_API_KEY`; it is not
exposed through the public orchestrator API.

Gemini suggestions are applied when they contain a non-empty changed value. Template-level
`validation_regex` rules remain in the extraction result as review signals for identifiers, dates,
and numeric fields; they do not block applying Gemini's suggestion. Names and handwriting should
normally remain reviewer-confirmed even when the VLM proposes a change.

## Tests

```bash
python -m pytest -q
```

The end-to-end tests cover registration, image processing, export, and a two-page PDF whose
fields are extracted from separate pages.
