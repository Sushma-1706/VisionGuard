import json
from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from .config import settings

class Base(DeclarativeBase): pass
class Inspection(Base):
    __tablename__ = "inspections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    prediction: Mapped[str] = mapped_column(String(32)); defect_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float); anomaly_score: Mapped[float] = mapped_column(Float)
    grounding_score: Mapped[float] = mapped_column(Float); hallucination_risk: Mapped[str] = mapped_column(String(16))
    explanation: Mapped[str] = mapped_column(Text); result_json: Mapped[str] = mapped_column(Text)

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
def init_database() -> None: Base.metadata.create_all(engine)
def save_inspection(record: Inspection) -> None:
    with SessionLocal() as session: session.add(record); session.commit()
def get_inspection(inspection_id: str) -> dict | None:
    with SessionLocal() as session:
        record = session.get(Inspection, inspection_id)
        return json.loads(record.result_json) if record else None
