import os
from sqlalchemy import create_engine, Column, String, Float, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None
Base = declarative_base()
db_available = False


class JobORM(Base):
    __tablename__ = "document_jobs"

    job_id = Column(String, primary_key=True, index=True)
    template_id = Column(String, index=True)
    status = Column(String, index=True)
    overall_confidence = Column(Float, default=0.0)
    needs_human_review = Column(Boolean, default=False)
    extracted_data_json = Column(Text, nullable=True)
    created_at = Column(String)
    completed_at = Column(String, nullable=True)


class TemplateORM(Base):
    __tablename__ = "templates"

    template_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    width = Column(Float, default=0.0)
    height = Column(Float, default=0.0)
    fields_json = Column(Text)


if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        db_available = True
        print(f"[Database] Successfully connected at {DATABASE_URL}")
    except Exception as e:
        print(f"[Database] Could not connect to database ({e}). Operating with in-memory store.")
