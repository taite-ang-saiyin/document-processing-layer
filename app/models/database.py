import os
from sqlalchemy import create_engine, Column, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None
Base = declarative_base()


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


if DATABASE_URL and "postgresql" in DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        print(f"[Database] Successfully connected to PostgreSQL at {DATABASE_URL}")
    except Exception as e:
        print(f"[Database] Could not connect to PostgreSQL ({e}). Operating with in-memory store.")
