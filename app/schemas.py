from pydantic import BaseModel
from typing import Optional, List

class UserProfileCreateFull(BaseModel):
    user_id: int
    interpretation: str
    id: float
    about: str
    portfolio: str
    skills: str
    specialization: str
    github: str
    github_tech_stack: str
    combined_text: str
    coordinator: str
    idea_generator: str
    evaluator: str
    collectivist: str
    perfectionist: str
    executor: str
    formulator: str
    specialist: str
    scout: str
    matched_specializations: str
     
    
class GetAllUserData(BaseModel):
    user_id: int
    interpretation: str
    id: float
    about: str
    portfolio: str
    skills: str
    specialization: str
    github: str
    github_tech_stack: str
    combined_text: str
    coordinator: str
    idea_generator: str
    evaluator: str
    collectivist: str
    perfectionist: str
    executor: str
    formulator: str
    specialist: str
    scout: str
    
class UpdateUserData(BaseModel):
    user_id: int
    matched_specializations: str
    
    
class TeamRequestBody(BaseModel):
    current_team: str
    order_type: str
    max_team_size: int
    
class ProjectRequestBody(BaseModel):
    order_type: str
    count: int