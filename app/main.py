from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from . import models, schemas, database
import pandas as pd
import requests
from pydantic import BaseModel
import os
from app.database import (
    engine, 
    SessionLocal, 
    Base, 
    insert_data, 
    get_all_data, 
    get_specialization_data, 
    convert_to_dataframe, 
    process_data, 
    build_team_to_project, 
    build_team,
    building_team,
    building_full_team,
    update_specializations,
    transform_user_profile_data,
    load_dotenv
)
from app.schemas import UserProfileCreateFull, TeamRequestBody, ProjectRequestBody
load_dotenv()
app = FastAPI(
    title="API для формирования команд",
    description="Простенькое API",
    version="1.0.0"
)

# Получение сессии БД
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/user/", include_in_schema=False)
async def get_user_profile_data(db: Session = Depends(get_db)):
    try:
        response = requests.post('https://actions.druid.1t.ru/webhook/subjects', json={'per_page': 100000, 'page': 1})

        response.raise_for_status()  # Проверяем успешность ответа
        api_data = response.json()
        print(api_data)
        transformed_profiles = transform_user_profile_data(api_data)
        inserted_profile = insert_data(transformed_profiles, db)
        return {"status": "success", "message": "Данные успешно внесены.", "inserted_data": inserted_profile}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Роут для получения всех данных
@app.get("/users/")
async def read_all_users(db: Session = Depends(get_db)):
    try:
        return get_all_data(db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Роут для получения данных с определённой специализацией
@app.get("/users/specializations/")
async def read_users_with_specialization(db: Session = Depends(get_db)):
    try:
        return get_specialization_data(db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Роут для обработки данных (назначение специализаций)
@app.post("/users/process/")
async def process_user_data(db: Session = Depends(get_db)):
    try:
        data = get_all_data(db)
        dataframe = convert_to_dataframe(data)
        processed_data = process_data(dataframe)
        update_specializations(processed_data, db)
        return {"status": "success", "message": "Данные успешно обработаны."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Роут для формирования команды проекта (POST запрос)
@app.post("/team/build_project/")
async def create_project_team(request_body: ProjectRequestBody, db: Session = Depends(get_db)):
    """
    order_type: тип заказа (определяет стратегию формирования команды)
    count: количество человек в команде
    """
    try:
        answer = building_team(db, request_body.order_type, request_body.count)
        return answer
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Роут для расширенного формирования команды (POST запрос)
@app.post("/team/build/")
async def build_team_route(request_body: TeamRequestBody, db: Session = Depends(get_db)):
    try:
        answer = building_full_team(db, request_body.order_type, request_body.max_team_size, request_body.current_team)
        return answer
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

