# API для формирования команд

## Описание
Этот API показывает примеры работы работы алгоритмов для формирования команд разработчиков. В нем реализовано формирование команд с нуля или с уже имеющимися участниками для определенной тематики задания. Полная информация по проекту находится на портале Druid.
#### Доступные тематики:
- Веб-сервис
- Аналитическая система
- Разработка 3D-игры
- Стартап по Data Science
- Создание корпоративного портала

## Используемые библиотеки
- FastAPI v0.68.0
- uvicorn v0.18.3
- SQLAlchemy v1.4.41
- Pandas v1.3.3
- NumPy v2.0.2
- requests v2.32.3
- Pydantic v1.8.2
- python-dotenv v1.0.1

## Установка
1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/SocialTech2024/1T_SocialTech.git
   ```

2. Перейти в каталог репозитория
   ```bash
   cd 1T_SocialTech
   ```
3. Запуск проекта возможен двумя способами.
- При помощи docker-compose 
  ```bash
   cd docker
   docker-compose -f app.docker-compose.yml up
   ```
- Ручной запуск(рекомендуется)
  ```bash
   pip install -r requirements.txt
   cd docker
   docker-compose -f db.docker-compose.yml up
   cd ..
   python -m uvicorn app.main:app --reload 
   ```

## Полная документация к API доступна по [ссылке](http://localhost:8000/docs)

