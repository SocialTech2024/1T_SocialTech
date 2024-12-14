from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json

from app.const_list import order_types

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class CandidateRecommender:
    def __init__(self, github_token, enable_git_parsing=True):
        # Инициализация модели и параметров
        self.tfidf_vectorizer = TfidfVectorizer()
        self.github_token = github_token
        self.enable_git_parsing = enable_git_parsing  # Управление парсингом GitHub

    # Парсинг GitHub, если включен флаг enable_git_parsing
    def parse_github(self, df):
        if not self.enable_git_parsing:
            print("Используются существующие данные.")
            return df

        def extract_github_username(url):
            if pd.isna(url) or 'github.com' not in url:
                return None
            return url.split('github.com/')[-1].split('/')[0]

        df['GitHub_Username'] = df['GitHub'].apply(extract_github_username)

        for index, row in df.iterrows():
            username = row['GitHub_Username']
            if username:
                print(f"Получение данных для пользователя: {username}")
                github_data = self.__get_github_data(username)
                if github_data:
                    for key, value in github_data.items():
                        df.at[index, key] = value
                time.sleep(1)

        return df

    # Приватный метод получения данных профиля пользователя GitHub
    def __get_github_data(self, username):
        headers = {'Authorization': f'token {self.github_token}'}
        base_url = f'https://api.github.com/users/{username}'
        try:
            profile_response = requests.get(base_url, headers=headers)
            if profile_response.status_code == 401:
                print("Ошибка 401: Неверный или истёкший токен GitHub.")
                return None
            elif profile_response.status_code == 404:
                print(f"Пользователь {username} не найден.")
                return None
            elif profile_response.status_code != 200:
                print(f"Ошибка {profile_response.status_code} при запросе профиля {username}")
                return None

            repos_url = f'{base_url}/repos'
            repos_response = requests.get(repos_url, headers=headers)
            if repos_response.status_code != 200:
                print(f"Ошибка {repos_response.status_code} при запросе репозиториев {username}")
                return None

            repos_data = repos_response.json()
            languages = set()
            for repo in repos_data:
                languages_url = repo.get('languages_url')
                if languages_url:
                    languages_response = requests.get(languages_url, headers=headers)
                    if languages_response.status_code == 200:
                        repo_languages = languages_response.json()
                        languages.update(repo_languages.keys())

            return {
                'GitHub_Languages': ', '.join(languages)
            }
        except requests.RequestException as e:
            print(f"Ошибка запроса для {username}: {e}")
            return None

    # Переименование колонок для унификации
    def rename_columns(self, df):   
        df = df.drop('Навыки (из справочника)', axis=1)
        return df.rename(columns={
            'ID': 'ID',
            'О себе': 'About',
            'Портфолио': 'Portfolio',
            'Навыки': 'Skills',
            'Специальность->Название': 'Specialization',
            'GitHub_Languages': 'GitHub_Tech_Stack'
        })
        
    def get_combined_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        def clean_text(text):
            if not text or not isinstance(text, str):
                return ''
            words = text.split()
            return ' '.join(word for word in words if word not in ['нет', 'информации', 'не', 'указано'])
          
        df['Combined_Text'] = (
            df['Specialization'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Skills'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['About'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Portfolio'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['GitHub_Tech_Stack'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x)
        ).str.lower()

        df['Combined_Text'] = df['Combined_Text'].apply(clean_text)
        
        return df
    
    def rename_column_df(safe, df):
        new_columns = {
        'about': 'About',
        'coordinator': 'Координатор',
        'scout': 'Разведчик',
        'id': 'ID',
        'idea_generator': 'Генератор идей',
        'matched_specializations': 'Matched_Specializations',
        'portfolio': 'Portfolio',
        'evaluator': 'Оценщик',
        'skills': 'Skills',
        'collectivist': 'Коллективист',
        'specialization': 'Specialization',
        'perfectionist': 'Доводчик',
        'user_id': 'User_ID',
        'github': 'GitHub',
        'executor': 'Реализатор',
        'github_tech_stack': 'GitHub_Tech_Stack',
        'formulator': 'Формирователь',
        'interpretation': 'interpretation',
        'combined_text': 'Combined_Text',
        'specialist': 'Специалист'
        }
        
        for column in ['id', 'user_id']:
            if column in df.columns:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    df[column] = df[column].astype(str)
                except ValueError as e:
                    safe.log(f"Ошибка преобразования колонки {column}: {str(e)}")
        df = df.rename(columns=new_columns)
        return df  
    
    def get_first_data(self, df):
        def clean_text(text):
            if not text or not isinstance(text, str):
                return ''
            words = text.split()
            return ' '.join(word for word in words if word not in ['нет', 'информации', 'не', 'указано'])
          
        df['Combined_Text'] = (
            df['Specialization'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Skills'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['About'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Portfolio'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['GitHub_Tech_Stack'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x)
        ).str.lower()

        df['Combined_Text'] = df['Combined_Text'].apply(clean_text)
        
        return df
    
    def get_data(self, df):
        return df
  
    # Подготовка данных
    def prepare_data(self, csv_file_path, excel_file_path):
        # Чтение данных из CSV и Excel
        df_csv = pd.read_csv(csv_file_path)
        df_excel = pd.read_excel(excel_file_path)

        #парсинг гита

        # Приведение идентификаторов к строковому формату для объединения
        df_csv['User_ID'] = df_csv['User_ID'].fillna(0).astype(int).astype(str).str.strip()
        df_excel['ID'] = df_excel['ID'].astype(str).str.strip()
        
        # Объединение данных по столбцам 'User_ID' и 'ID'
        df = pd.merge(
            # df_csv[['User_ID', 'name', 'interpretation']],
            df_csv[['User_ID', 'interpretation']],
            df_excel,
            left_on='User_ID',
            right_on='ID', 
            how='outer'
        )

        df = self.parse_github(df)
        
        df = self.rename_columns(df)        
        # Очистка текста в объединенных данных
        def clean_text(text):
            if not text or not isinstance(text, str):
                return ''
            words = text.split()
            return ' '.join(word for word in words if word not in ['нет', 'информации', 'не', 'указано'])

        # df = df.drop('ID', axis=1)
        
        if 'GitHub_Tech_Stack' not in df.columns:
            df['GitHub_Tech_Stack'] = ''

        df['Combined_Text'] = (
            df['Specialization'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Skills'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['About'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['Portfolio'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x) + ' ' +
            df['GitHub_Tech_Stack'].fillna('').astype(str).apply(lambda x: '' if x.strip().lower() in ['не указано', 'нет информации'] else x)
        ).str.lower()

        df['Combined_Text'] = df['Combined_Text'].apply(clean_text)

        return df
    
    # Генерация текстов для специальностей с кастомными весами
    def generate_specialization_texts(self, specializations):
        texts = []
        for spec, skills in specializations.items():
            spec_name = ' '.join([spec.lower()] * 10)
            primary_skills = ' '.join([skill.lower() for skill in skills["primary"]] * 5)
            secondary_skills = ' '.join([skill.lower() for skill in skills["secondary"]] * 3)
            tertiary_skills = ' '.join([skill.lower() for skill in skills["tertiary"]])
            texts.append(f"{spec_name} {primary_skills} {secondary_skills} {tertiary_skills}")
        return texts


    def add_belbin_roles(self, df):
        def parse_belbin_roles(interpretation):
            try:
                roles = json.loads(interpretation) if isinstance(interpretation, str) else []
                return [{"name": role["name"], "score": role["score"]} for role in roles]
            except Exception as e:
                print(f"Parsing error: {e}, data: {interpretation}")
                return []

        def calculate_top_roles_with_percentages(roles):
            if not roles or not isinstance(roles, list):
                return {}
            
            sorted_roles = sorted(roles, key=lambda x: x["score"], reverse=True)[:3]
            total_score = sum(role["score"] for role in sorted_roles)
            if total_score == 0:
                return {role["name"]: 0 for role in roles}
            
            top_roles_percentages = {
                role["name"]: round((role["score"] / total_score) * 100, 2)
                for role in sorted_roles
            }
            
            all_roles = [
                "Координатор", "Генератор идей", "Оценщик", "Коллективист",
                "Доводчик", "Реализатор", "Формирователь", "Специалист", "Разведчик"
            ]
            return {role: top_roles_percentages.get(role, 0) for role in all_roles}

        role_columns = [
            "Координатор", "Генератор идей", "Оценщик", "Коллективист",
            "Доводчик", "Реализатор", "Формирователь", "Специалист", "Разведчик"
        ]

        df['parsed_roles'] = df['interpretation'].apply(parse_belbin_roles)

        for role in role_columns:
            df[role] = df['parsed_roles'].apply(
                lambda roles: calculate_top_roles_with_percentages(roles).get(role, 0)
            )

        df.drop(columns=['parsed_roles'], inplace=True)
        return df
      
    # Поиск кандидатов
    def find_candidates(self, candidates_df, specializations, input_data=None):
        # Нормализуем справочник специализаций
        
        def normalize_specializations(specializations):
            normalized = {}
            for spec, skills in specializations.items():
                normalized[spec.lower()] = {
                    "primary": [skill.lower() for skill in skills["primary"]],
                    "secondary": [skill.lower() for skill in skills["secondary"]],
                    "tertiary": [skill.lower() for skill in skills["tertiary"]],
                }
            return normalized
        
        specializations = normalize_specializations(specializations)

        if input_data is None:
            specializations_list = list(specializations.keys())
            specialization_texts = self.generate_specialization_texts(specializations)

            # Нормализуем текст кандидатов
            combined_texts = candidates_df['Combined_Text'].fillna('').str.lower().tolist()
            all_texts = specialization_texts + combined_texts

            # Вычисляем TF-IDF матрицу
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            specialization_matrix = tfidf_matrix[:len(specializations_list)]
            candidate_matrix = tfidf_matrix[len(specializations_list):]

            similarity_matrix = cosine_similarity(candidate_matrix, specialization_matrix)

            matched_specializations = []
            for candidate_index, similarities in enumerate(similarity_matrix):
                # Усиливаем совпадение по названию специальности
                for idx, spec in enumerate(specializations_list):
                    if spec in combined_texts[candidate_index]:  # Сравнение уже нормализовано
                        similarities[idx] += 1.5

                # Усиливаем совпадение по навыкам
                for idx, spec in enumerate(specializations_list):
                    skills = specializations[spec]
                    primary_matches = sum(skill in combined_texts[candidate_index] for skill in skills["primary"])
                    secondary_matches = sum(skill in combined_texts[candidate_index] for skill in skills["secondary"])
                    tertiary_matches = sum(skill in combined_texts[candidate_index] for skill in skills["tertiary"])

                    similarities[idx] += primary_matches * 2.0  # Усиление веса первичных навыков
                    similarities[idx] += secondary_matches * 1.0
                    similarities[idx] += tertiary_matches * 0.5

                # Сортируем совпадения по убыванию
                sorted_matches = sorted(
                    zip(specializations_list, similarities),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Ограничиваемся топ-3 совпадениями с порогом > 0.2
                filtered_matches = [
                    f"{spec} ({round(score, 4)})" for spec, score in sorted_matches if score > 0.2
                ][:3]

                # Если ничего не найдено, добавляем "Нет подходящей специальности"
                if not filtered_matches:
                    filtered_matches = ["Нет подходящей специальности"]

                matched_specializations.append(', '.join(filtered_matches))

            candidates_df['Matched_Specializations'] = matched_specializations
            return candidates_df

        elif isinstance(input_data, list):
            # Обработка поиска по навыкам
            combined_texts = candidates_df['Combined_Text'].fillna('').tolist()
            all_texts = [' '.join(input_data)] + combined_texts

            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            user_vector = tfidf_matrix[0]
            candidate_matrix = tfidf_matrix[1:]

            # Вычисление косинусного сходства
            similarities = cosine_similarity(user_vector, candidate_matrix).flatten()
            candidates_df['Similarity'] = similarities
            return candidates_df.sort_values(by='Similarity', ascending=False)

        else:
            raise ValueError("Входными данными может быть список навыков либо название специализации.")

    # Получение топ-N кандидатов
    def find_top_candidates_by_specialization_from_results(self, candidates_df, specializations_str, top_n=5):
        print(candidates_df.columns)
        if 'Matched_Specializations' not in candidates_df.columns:
            raise ValueError("Колонка 'Matched_Specializations' отсутствует в DataFrame. Убедитесь, что find_candidates выполнен.")

        required_specializations = [spec.strip().lower() for spec in specializations_str.split(',')]

        def calculate_relevance(specializations_str):
            if not isinstance(specializations_str, str):
                return 0, 0 
            
            matches = [match.strip() for match in specializations_str.split(',')]
            match_count = 0
            max_similarity = 0
            
            for match in matches:
                for specialization in required_specializations:
                    if specialization in match:
                        try:
                            similarity = float(match.split('(')[-1].strip(')'))
                            max_similarity = max(max_similarity, similarity)
                            match_count += 1
                        except ValueError:
                            continue
            return match_count, max_similarity

        
        candidates_df['Match_Count'], candidates_df['Max_Similarity'] = zip(
            *candidates_df['Matched_Specializations'].apply(calculate_relevance)
        )

        candidates_df = candidates_df.sort_values(
            by=['Match_Count', 'Max_Similarity'], 
            ascending=[False, False]
        )

        return candidates_df.head(top_n).drop(columns=['Match_Count', 'Max_Similarity'], errors='ignore')
    
    # Формирование команды под указанный тип заказа / проекта)
    def build_team_for_project(self, candidates_df, order_type, team_size=5):
       
        if order_type not in order_types:
            raise ValueError(f"Тип заказа '{order_type}' отсутствует в справочнике.")

        required_specializations = order_types[order_type]
        team = []
        remaining_candidates = candidates_df.copy()

        # Подбор кандидатов для каждой специализации
        for specialization in required_specializations:
            if len(team) >= team_size:
                break

            # Находим кандидатов для текущей специализации
            candidates_for_specialization = self.find_top_candidates_by_specialization_from_results(
                remaining_candidates,
                specialization,
                top_n=1  
            )

            if not candidates_for_specialization.empty:
                selected_candidate = candidates_for_specialization.iloc[0].to_dict()  
                team.append(selected_candidate)
                remaining_candidates = remaining_candidates[remaining_candidates['ID'] != selected_candidate['ID']]
        # В случае, если указано количество человек больше, чем специализаций в заказе, то 
        # добираем менее релевантных кандидатов
        if len(team) < team_size:
            # Вычисляем среднюю релевантность кандидатов к задачам заказа
            def calculate_relevance(row):
                total_relevance = 0
                for specialization in required_specializations:
                    relevance = self.calculate_specialization_relevance(row['Matched_Specializations'], specialization)
                    total_relevance += relevance
                return total_relevance

            remaining_candidates['Relevance_To_Order'] = remaining_candidates.apply(calculate_relevance, axis=1)

            # Сортируем оставшихся кандидатов по релевантности
            additional_candidates = remaining_candidates.sort_values(by='Relevance_To_Order', ascending=False)
            
            # Добираем недостающих кандидатов
            additional_candidates = additional_candidates.head(team_size - len(team)).to_dict('records')
            team.extend(additional_candidates)

        return pd.DataFrame(team)

    # Функция для вычисления релевантности кандидата к специализации.
    def calculate_specialization_relevance(self, matched_specializations, specialization):
        if not isinstance(matched_specializations, str):
            return 0
        
        matches = [match.strip() for match in matched_specializations.split(',')]
        for match in matches:
            if specialization.lower() in match.lower():
                try:
                    return float(match.split('(')[-1].strip(')'))
                except ValueError:
                    return 0
        return 0
    

    # Подбор кандидатов по специализациям и ролям Белбина с сортировкой по релевантности.
    def build_team_by_specialization_and_belbin(self, candidates_df, current_team, order_type, max_team_size=20):
        
        if not isinstance(current_team, pd.DataFrame):
            raise ValueError("current_team должен быть DataFrame.")

        # Создаем копии DataFrame
        candidates_df = candidates_df.copy()
        current_team = current_team.copy()

        # Приведение User_ID к строкам
        candidates_df['User_ID'] = candidates_df['User_ID'].astype(str).str.strip()
        current_team['User_ID'] = current_team['User_ID'].astype(str).str.strip()

        # Получаем список текущих ID
        current_ids = current_team['User_ID'].tolist()

        # Оставшиеся кандидаты
        remaining_candidates = candidates_df[~candidates_df['User_ID'].isin(current_ids)].copy()

        # Проверка и приведение типов данных в remaining_candidates
        for col in remaining_candidates.columns:
            if remaining_candidates[col].dtype == 'object':
                try:
                    remaining_candidates[col] = pd.to_numeric(remaining_candidates[col], errors='coerce')
                except Exception as e:
                    raise ValueError(f"Ошибка приведения данных в столбце '{col}': {e}")

        # Формируем команду по специализации
        try:
            specialization_team = self.build_team_for_project(remaining_candidates, order_type, team_size=max_team_size)
        except Exception as e:
            raise ValueError(f"Ошибка в build_team_for_project: {e}")

        # Объединяем команды
        extended_team = pd.concat([current_team, specialization_team], ignore_index=True)
        extended_team = extended_team.drop_duplicates(subset='User_ID', keep='first')

        # Обновляем оставшихся кандидатов
        remaining_candidates = candidates_df[~candidates_df['User_ID'].isin(current_ids)].copy()
        # Приведение типов данных в extended_team перед балансировкой
        for col in extended_team.columns:
            if extended_team[col].dtype == 'object':
                try:
                    extended_team[col] = pd.to_numeric(extended_team[col], errors='coerce')
                except Exception as e:
                    raise ValueError(f"Ошибка приведения данных в столбце '{col}' extended_team: {e}")

        # Балансируем команду
        try:
            balanced_team = self.balance_team_by_belbin(extended_team, remaining_candidates, max_team_size, current_ids)
        except Exception as e:
            raise ValueError(f"Ошибка в balance_team_by_belbin: {e}")

        return balanced_team
    
    
    
    
    # Балансировка команды по тесту Белбина
    def balance_team_by_belbin(self, extended_team, remaining_candidates, max_team_size, current_team_ids):
        role_columns = [
            "Координатор", "Генератор идей", "Оценщик", "Коллективист",
            "Доводчик", "Реализатор", "Формирователь", "Специалист", "Разведчик"
        ]
        
        # Проверяем наличие столбцов и заполняем отсутствующие
        for col in role_columns:
            if col not in extended_team.columns:
                extended_team[col] = 0
            if col not in remaining_candidates.columns:
                remaining_candidates[col] = 0

        # Приведение данных к числовому типу с заполнением NaN
        extended_team[role_columns] = extended_team[role_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        remaining_candidates[role_columns] = remaining_candidates[role_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Текущие роли в команде
        current_roles = extended_team[role_columns].sum()

        # Расчет недостающих ролей
        avg_role_count = max_team_size / len(role_columns)
        missing_roles = {role: max(0, int(avg_role_count - current_roles[role])) for role in role_columns}

        for role, count in missing_roles.items():
            if count > 0:
                role_candidates = remaining_candidates[remaining_candidates[role] > 0].copy()
                role_candidates = role_candidates.sort_values(by=role, ascending=False)

                for _ in range(count):
                    if role_candidates.empty:
                        break
                    selected_candidate = role_candidates.iloc[0]
                    
                    # Заполняем отсутствующие столбцы для выбранного кандидата
                    for col in extended_team.columns:
                        if col not in selected_candidate.index:
                            selected_candidate[col] = 0
                    
                    # Добавляем кандидата в команду
                    extended_team = pd.concat(
                        [extended_team, pd.DataFrame([selected_candidate.to_dict()])], ignore_index=True
                    )
                    
                    # Удаляем кандидата из оставшихся
                    remaining_candidates = remaining_candidates[
                        remaining_candidates['User_ID'] != selected_candidate['User_ID']
                    ]

        # Убираем текущих участников команды из итогового списка
        extended_team = extended_team[~extended_team['User_ID'].isin(current_team_ids)]

        # Убедимся, что NaN отсутствуют в итоговой команде
        extended_team = extended_team.fillna(0)
        return extended_team.head(max_team_size)
