from sqlalchemy import Column, Integer, String, Float, Text
from app.database import Base

class ProcessedData(Base):
    __tablename__ = "processed_data"

    user_id = Column(Integer, primary_key=True, index=True)
    interpretation = Column(Text)
    id = Column(Float)
    about = Column(Text)
    portfolio = Column(Text)
    skills = Column(Text)
    specialization = Column(Text)
    github = Column(Text)
    github_tech_stack = Column(Text)
    combined_text = Column(Text)
    coordinator = Column(Text)
    idea_generator = Column(Text)
    evaluator = Column(Text)
    collectivist = Column(Text)
    perfectionist = Column(Text)
    executor = Column(Text)
    formulator = Column(Text)
    specialist = Column(Text)
    scout = Column(Text)
    matched_specializations = Column(Text)
