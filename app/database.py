from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm import sessionmaker, Session
from fastapi import HTTPException
from . import models, schemas
from app.const_list import github_token, specializations
from app.classes import *

DATABASE_URL = "postgresql://postgres:123456@localhost/1t"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def update_specializations(user_profiles: list, db: Session):
    try:
        # Начинаем транзакцию
        for user_profile in user_profiles:
            # Проверяем, существует ли профиль пользователя в базе данных
            db_user_profile = db.query(models.ProcessedData).filter(models.ProcessedData.user_id == user_profile['user_id']).first()

            if not db_user_profile:
                raise HTTPException(status_code=404, detail=f"Пользователь с ID {user_profile['user_id']} не найден")

            # Обновляем специализации пользователя
            db_user_profile.matched_specializations = user_profile['matched_specializations']

            # Сохраняем изменения для текущего пользователя
            db.commit()
            db.refresh(db_user_profile)

        return {"status": "success", "message": "Данные успешно обновлены."}

    except SQLAlchemyError as e:
        db.rollback()  # Откатываем транзакцию на случай ошибки
        raise HTTPException(status_code=500, detail=str(e))
    

def insert_data(user_profiles: list, db: Session):
    inserted_profiles = []

    for user_profile in user_profiles:
        # Проверка типа объекта перед вставкой
        if not isinstance(user_profile, dict):
            print(f"Недопустимый объект: {type(user_profile)}")
            continue
        
        # Проверка обязательных данных
        if not user_profile.get('user_id') or not user_profile.get('skills') or not user_profile.get('specialization'):
            print(f"Недопустимый профиль: {user_profile}")
            continue

        try:
            # Создание объекта для вставки в базу данных
            db_user_profile = models.ProcessedData(
                user_id=user_profile['user_id'],
                interpretation=user_profile['interpretation'],
                about=user_profile['about'],
                portfolio=user_profile['portfolio'],
                skills=user_profile['skills'],
                specialization=user_profile['specialization']
            )
            db.add(db_user_profile)
            db.commit()  # Commit the transaction

            # Refresh to get the inserted object with the correct ID
            db.refresh(db_user_profile)
            inserted_profiles.append(db_user_profile)
        except Exception as e:
            print(f"Ошибка при добавлении профиля: {e}")
            db.rollback()  # Rollback the transaction to avoid partial inserts
            raise HTTPException(status_code=500, detail=str(e))  # Raise HTTPException for error reporting

    return inserted_profiles


# Функция для получения всех записей из базы данных
def get_all_data(db: Session):
    try:
        # Выполняем запрос к базе данных
        user_profiles = db.query(models.ProcessedData).all()
        
        # Проверяем, есть ли данные
        if not user_profiles:
            raise ValueError("Запрос к базе данных не вернул данных.")
        
        return user_profiles
    except Exception as e:
        raise ValueError(f"Ошибка получения данных: {e}")

def transform_user_profile_data(api_data: dict):
    """
    Преобразование данных из API для дальнейшего внесения в базу данных.

    :param api_data: Данные, полученные из стороннего API.
    :return: Список преобразованных профилей пользователей.
    """
    transformed_profiles = []

    # Проверка наличия ключей в данных API
    api_profiles = api_data.get('data', {}).get('paginate_subject', {}).get('data', [])

    if not api_profiles or not isinstance(api_profiles, list):
        print("Ошибка: Данные из API не содержат ожидаемых профилей или они пустые.")
        return transformed_profiles

    for api_profile in api_profiles:
        try:
            # Проверка на наличие правильных данных в каждом профиле
            if not isinstance(api_profile, dict):
                print(f"Ошибка: Ожидался словарь, а получено {type(api_profile)}")
                continue

            # Преобразование списка навыков в строку
            skills = ", ".join(
                skill.get('object', {}).get('name', 'Не указано')
                for skill in (api_profile.get('skills') or [])
                if skill is not None
            )

            # Преобразование списка интерпретаций в строку
            interpretation = ", ".join(
                interp.get('name', 'Не указано')
                for interp in (api_profile.get('interpretation') or [])
                if interp is not None
            )

            # Создание словаря для одного профиля пользователя
            transformed_profiles.append({
                "user_id": int(api_profile.get('id', 0)),  # ID пользователя
                "about": api_profile.get('about', ''),
                "specialization": (
                    (api_profile.get('speciality1') or {}).get('object', {}).get('name', 'Не указано')
                ),
                "skills": skills,
                "interpretation": interpretation,
                "matched_specializations": api_profile.get('matched_specializations', 'Нет подходящей специальности'),
                "portfolio": api_profile.get('portfolio', []),
                "region": api_profile.get('region', ''),
                "experience": api_profile.get('experience', ''),
                "github": api_profile.get('github', None)
            })
        except Exception as e:
            print(f"Ошибка при обработке профиля: {e}")
            continue

    return transformed_profiles

# Функция для получения записей, у которых определена специализация
def get_specialization_data(db: Session):
    try:
        return db.query(models.ProcessedData).filter(
            models.ProcessedData.matched_specializations != "Нет подходящей специальности"
        ).all()
    except Exception as e:
        raise ValueError(f"Ошибка получения данных с фильтром: {e}")

# Функция для конвертации тела запроса в DataFrame
def convert_to_dataframe(data: list):
    try:
        if not data:
            raise ValueError("Нет данных для преобразования в DataFrame.")
        # Преобразуем объекты в DataFrame
        dataframe = pd.DataFrame([profile.__dict__ for profile in data])
        
        # Проверяем, что DataFrame не пустой
        if dataframe.empty:
            raise ValueError("DataFrame пуст.")
        
        # Удаляем системную колонку, если она существует
        if "_sa_instance_state" in dataframe.columns:
            dataframe.drop(columns="_sa_instance_state", inplace=True)
        
        return dataframe
    except ValueError as e:
        raise ValueError(f"Ошибка преобразования данных в DataFrame: {e}")
    
def format_answer(current_team_formated, final_team):
    # Преобразуем типы данных для User_ID в строку
    current_team_formated["User_ID"] = current_team_formated["User_ID"].astype(float)
    final_team["User_ID"] = final_team["User_ID"].astype(float)

    # Проверяем, что необходимые столбцы существуют в current_team_formated
    required_columns = ["User_ID", "interpretation", "Matched_Specializations"]
    for col in required_columns:
        if col not in current_team_formated.columns:
            raise ValueError(f"В current_team_formated отсутствует столбец '{col}'")

    # Проверяем наличие столбца User_ID в final_team
    if "User_ID" not in final_team.columns:
        raise ValueError("В final_team отсутствует столбец 'User_ID'.")


    # Фильтруем строки по User_ID
    result = current_team_formated[current_team_formated["User_ID"].isin(final_team["User_ID"])]

    # Если результат пустой, выбрасываем исключение
    if result.empty:
        raise ValueError("Фильтрация по User_ID не дала результатов. Проверьте данные.")

    # Возвращаем только нужные колонки
    return result[required_columns]



# Функция для назначения специализации, работает с DataFrame
def process_data(dataframe: pd.DataFrame):
    try:
        recommender = CandidateRecommender(github_token=github_token, enable_git_parsing=False)
        df = recommender.get_data(dataframe)
        df = recommender.rename_column_df(df)
        data = recommender.find_candidates(df, specializations)
        answer  = convert_to_json_answer(data)
        return answer
    except Exception as e:
        raise ValueError(f"Ошибка обработки данных: {e}")

# Функция подбора команды для проекта, работает с DataFrame
def build_team_to_project(team_df, count, order_type):
    try:
        if team_df.empty:
            raise ValueError("Передан пустой DataFrame для формирования команды.")
        recommender = CandidateRecommender(github_token=github_token, enable_git_parsing=False)
        df_renaimed = recommender.rename_column_df(team_df)
        team = recommender.build_team_for_project(df_renaimed, order_type, count)
        return team[["User_ID", "interpretation", "Matched_Specializations"]]
    except Exception as e:
        raise ValueError(f"Ошибка формирования команды: {type(e).__name__}, {e}")

def current_team_format(current_team, candidates_df):
    try:
        # Проверяем, что current_team это строка и разделяем её на user_id
        if isinstance(current_team, str):
            team_ids = list(map(str, current_team.split(',')))
        elif isinstance(current_team, list) and all(isinstance(item, str) for item in current_team):
            team_ids = current_team
        else:
            raise ValueError("current_team должно быть либо строкой с ID через запятую, либо списком целых чисел.")

        # Фильтруем DataFrame по user_id
        filtered_candidates = candidates_df[candidates_df['User_ID'].isin(team_ids)]
        return filtered_candidates

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



# Функция для формирования команды, работает с DataFrame
def build_team(team_df, current_team, order_type, max_team_size):
    try:
        # Проверка на пустой DataFrame
        if team_df.empty:
            raise ValueError("Один из переданных DataFrame пуст.")
        
        # Инициализация рекомендательной системы
        recommender = CandidateRecommender(github_token=github_token, enable_git_parsing=False)
        
        # Переименование столбцов
        candidates_df = recommender.rename_column_df(team_df)
        # Проверка наличия столбца 'User_ID' после переименования
        if 'User_ID' not in candidates_df.columns:
            raise KeyError("После переименования в candidates_df отсутствует столбец 'User_ID'.")
        
        # Форматирование текущей команды
        current_team_formated = current_team_format(current_team, candidates_df)
        # Построение финальной команды
        final_team = recommender.build_team_by_specialization_and_belbin(
            candidates_df=candidates_df,
            current_team=current_team_formated,
            order_type=order_type,
            max_team_size=max_team_size
        )
        # Форматирование результата
        result = format_answer(candidates_df, final_team)

        return result
    
    except KeyError as ke:
        raise ValueError(f"build_team Ошибка формирования команды: KeyError, {ke}")
    except ValueError as ve:
        raise ValueError(f"build_team Ошибка формирования команды: ValueError, {ve}")
    except Exception as e:
        raise ValueError(f"build_team Ошибка формирования команды: {type(e).__name__}, {e}")


def building_team(db: Session, order_type, count):
    try:
        data = get_specialization_data(db)
        team_df = convert_to_dataframe(data)
        final_team = build_team_to_project(team_df, count, order_type)
        answer = convert_to_json_answer(final_team)
        return answer
    except Exception as e:
        raise ValueError(f"Ошибка формирования команды: {type(e).__name__}, {e}")


def building_full_team(db: Session, order_type, max_team_size, current_team):
    try:
        data = get_specialization_data(db)
        team_df = convert_to_dataframe(data)
        final_team = build_team(team_df, current_team, order_type, max_team_size)
        answer = convert_to_json_answer(final_team)
        return answer
    except Exception as e:
        raise ValueError(f"Ошибка формирования команды: {type(e).__name__}, {e}")


def convert_to_json_answer(df):
    grouped_data = []
    for index, row in df.iterrows():
        grouped_entry = {
            "user_id": int(row['User_ID']),
            "matched_specializations": row['Matched_Specializations']
        }
        grouped_data.append(grouped_entry)
        
    return grouped_data

