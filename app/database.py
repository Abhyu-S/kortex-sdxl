"""
Database configuration for Kortex task engine.

Provides SQLAlchemy engine, session factory, and declarative base,
all wired to the Postgres instance defined in docker-compose.yml.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://kortex:kortex_secret@db:5432/kortex_tasks",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
