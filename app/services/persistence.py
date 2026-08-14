from typing import Dict

from app.models import database
from app.models.schemas import TemplateDefinition, DocumentProcessingJob


class PersistenceService:
    """Persists templates and processing jobs to the database when available.

    Always mirrors state to the in-memory registries so the runtime behaves the
    same whether a database is configured or not. When a database connection is
    unavailable (e.g. local dev without Postgres), it degrades gracefully to the
    in-memory store used previously.
    """

    def template_exists(self, template_id: str, registry: Dict[str, TemplateDefinition]) -> bool:
        if template_id in registry:
            return True
        if not database.db_available:
            return False
        try:
            with database.SessionLocal() as session:
                return session.get(database.TemplateORM, template_id) is not None
        except Exception as exc:
            print(f"[Persistence] Failed to check template '{template_id}': {exc}")
            return False

    def save_template(self, template: TemplateDefinition, registry: Dict[str, TemplateDefinition]) -> None:
        registry[template.template_id] = template
        if not database.db_available:
            return
        try:
            with database.SessionLocal() as session:
                row = database.TemplateORM(
                    template_id=template.template_id,
                    name=template.name,
                    width=template.width,
                    height=template.height,
                    fields_json=template.model_dump_json(),
                )
                session.merge(row)
                session.commit()
        except Exception as exc:
            print(f"[Persistence] Failed to persist template '{template.template_id}': {exc}")

    def save_job(self, job: DocumentProcessingJob, jobs_store: Dict[str, DocumentProcessingJob]) -> None:
        jobs_store[job.job_id] = job
        if not database.db_available:
            return
        try:
            with database.SessionLocal() as session:
                row = database.JobORM(
                    job_id=job.job_id,
                    template_id=job.template_id,
                    status=job.status.value,
                    overall_confidence=job.overall_confidence,
                    needs_human_review=job.needs_human_review,
                    extracted_data_json=job.model_dump_json(),
                    created_at=job.created_at,
                    completed_at=job.completed_at,
                )
                session.merge(row)
                session.commit()
        except Exception as exc:
            print(f"[Persistence] Failed to persist job '{job.job_id}': {exc}")

    def load_templates(self, registry: Dict[str, TemplateDefinition]) -> None:
        if not database.db_available:
            return
        try:
            with database.SessionLocal() as session:
                for row in session.query(database.TemplateORM).all():
                    try:
                        template = TemplateDefinition.model_validate_json(row.fields_json)
                        registry[template.template_id] = template
                    except Exception as exc:
                        print(f"[Persistence] Skipping invalid template row '{row.template_id}': {exc}")
        except Exception as exc:
            print(f"[Persistence] Failed to load templates: {exc}")

    def load_jobs(self, jobs_store: Dict[str, DocumentProcessingJob]) -> None:
        if not database.db_available:
            return
        try:
            with database.SessionLocal() as session:
                for row in session.query(database.JobORM).all():
                    try:
                        job = DocumentProcessingJob.model_validate_json(row.extracted_data_json)
                        jobs_store[job.job_id] = job
                    except Exception as exc:
                        print(f"[Persistence] Skipping invalid job row '{job.job_id}': {exc}")
        except Exception as exc:
            print(f"[Persistence] Failed to load jobs: {exc}")