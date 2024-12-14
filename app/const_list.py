order_types = {
    "Веб-сервис": [
        "Фронтенд-разработчик", 
        "Бэкенд-разработчик", 
        "Фулстек-разработчик", 
        "Дизайнер", 
        "DevOps инженер"
    ],
    "Аналитическая система": [
        "Аналитик данных", 
        "Специалист по Data Science", 
        "Инженер данных", 
        "Системный аналитик"
    ],
    "Разработка 3D-игры": [
        "Разработчик 3D-игр на языке JS", 
        "Дизайнер", 
        "Фулстек-разработчик", 
        "Тестировщик"
    ],
    "Стартап по Data Science": [
        "Специалист по Data Science", 
        "Инженер данных", 
        "DevOps инженер", 
        "Бизнес-аналитик", 
        "Менеджер"
    ],
    "Создание корпоративного портала": [
        "Системный аналитик", 
        "Бизнес-аналитик", 
        "Фулстек-разработчик", 
        "Тестировщик", 
        "Дизайнер"
    ],
}


specializations = {
    "Менеджер": {
        "primary": ["Управление проектом", "Agile", "Scrum", "Kanban", "Jira"],
        "secondary": ["Управление персоналом", "Бизнес-процесс", "Confluence", "MS Project", "PMBoK"],
        "tertiary": ["Тайм-менеджмент", "Проведение презентации", "Google Docs", "Notion", "Разработка концепции"]
    },
    "Аналитик данных": {
        "primary": ["SQL", "Python", "Анализ данных", "Power BI", "Excel"],
        "secondary": ["База данных", "PostgreSQL", "ETL", "Tableau", "BI"],
        "tertiary": ["NumPy", "A/B тест", "Математический анализ", "Pandas", "Scikit-learn"]
    },
    "Специалист по 1С": {
        "primary": ["1С", "1С: Предприятие", "1С: Бухгалтерия", "Консультирование пользователей"],
        "secondary": ["1С: Зарплата и управление персоналом", "1С: Документооборот", "Анализ бизнес-процесса"],
        "tertiary": ["Документооборот", "Техническая поддержка"]
    },
    "Тестировщик": {
        "primary": ["SQL", "API", "REST", "Ручное тестирование", "Функциональное тестирование"],
        "secondary": ["Jira", "Postman", "Тестирование мобильного приложения", "Автоматизация тестирования", "Selenium"],
        "tertiary": ["HTTP", "JSON", "Тестирование API", "Интеграционное тестирование", "Регрессионное тестирование"]
    },
    "Специалист по Data Science": {
        "primary": ["Python", "Machine learning", "SQL", "PyTorch", "NLP"],
        "secondary": ["NumPy", "TensorFlow", "Scikit-learn", "Pandas", "CatBoost"],
        "tertiary": ["Big Data", "Computer vision", "XGBoost", "Deep learning", "OpenCV"]
    },
    "Маркетолог": {
        "primary": ["СustDev", "SEO", "CJM", "Яндекс Директ", "Реклама"],
        "secondary": ["SMM", "Анализ рынка", "Инструменты маркетинга и рекламы Google", "Копирайтинг", "Power BI"],
        "tertiary": ["Маркетинговая стратегия", "E-commerce", "Email-маркетинг", "MyTarget", "B2B-marketing"]
    },
    "Фронтенд-разработчик": {
        "primary": ["JavaScript", "TypeScript", "React", "CSS", "HTML"],
        "secondary": ["Vue.js", "API", "Webpack", "REST", "Redux"],
        "tertiary": ["Sass", "Docker", "JQuery", "Bootstrap", "Node.js"]
    },
    "Системный аналитик": {
        "primary": ["SQL", "REST", "UML", "BPMN", "Jira"],
        "secondary": ["Confluence", "Agile", "Scrum", "Kanban", "ER-model"],
        "tertiary": ["Visio", "JSON", "XML", "SoapUI", "Flowchart"]
    },
    "Специалист по БАС": {
        "primary": ["Python", "SITL", "Ardupilot", "Робототехника", "Embedded", "Arduino", "OrangePi", "RaspberryPi"],
        "secondary": ["QGroundControl", "ROS", "SolidWorks", "Mission Planner", "3D-моделирование"],
        "tertiary": ["Авиастроение", "Pixhawk", "КОМПАС-3D", "3D-печать", "C/C++"]
    },
    "DevOps инженер": {
        "primary": ["Linux", "CI/CD", "Docker", "Kubernetes", "Ansible"],
        "secondary": ["Grafana", "Prometheus", "Git", "Jenkins", "Terraform"],
        "tertiary": ["Redis", "MongoDB", "Hadoop", "Airflow", "Kafka"]
    },
    "Дизайнер": {
        "primary": ["UX", "Figma", "UI", "Adobe Photoshop", "Adobe Illustrator"],
        "secondary": ["Web design", "Tilda", "Adobe XD", "Sketch", "Axure RP"],
        "tertiary": ["Miro", "Landing page", "Дизайн полиграфии", "Adobe After Effects", "Брендинг"]
    },
    "Инженер данных": {
        "primary": ["SQL", "Python", "ETL", "PostgreSQL", "Airflow"],
        "secondary": ["Spark", "Kafka", "ClickHouse", "MS SQL", "Oracle"],
        "tertiary": ["Greenplum", "MongoDB", "Redis", "Superset", "DWH"]
    },
    "Фулстек-разработчик": {
        "primary": ["JavaScript", "Python", "SQL", "React", "PostgreSQL"],
        "secondary": ["Node.js", "REST", "MongoDB", "Docker", "HTML"],
        "tertiary": ["GraphQL", "SOAP", "Bootstrap", "MySQL", "FastAPI"]
    },
    "Бэкенд-разработчик": {
        "primary": ["Python", "Docker", "REST", "PostgreSQL", "SQL"],
        "secondary": ["RabbitMQ", "Kafka", "MongoDB", "FastAPI", "Kubernetes"],
        "tertiary": ["Elasticsearch", "SOAP", "Redis", "ClickHouse", "Django"]
    },
    "Бизнес-аналитик": {
        "primary": ["SQL", "BPMN", "BI", "Power BI", "UML"],
        "secondary": ["Jira", "Confluence", "Бизнес-процесс", "SWOT-анализ", "Agile"],
        "tertiary": ["DFD", "Flowchart", "ERD", "ИDEF", "Business studio"]
    },
    "Разработчик 3D-игр на языке JS": {
        "primary": ["Babylon JS", "Оформление кода в объекты и классы", "Знание 3D движков", "Основы линейной алгебры", "ООП", "JavaScript", "TypeScript", "HTML", "CSS"],
        "secondary": ["Создание интерактивных 3D-сцен", "Оптимизация производительности игр", "Интеграция WebGL", "Работа с текстурами и шейдерами", "Обработка пользовательского ввода"],
        "tertiary": ["Работа с физическими движками", "Проектирование игровых уровней", "Коллаборация с дизайнерами и художниками", "Деплой веб-приложений", "Тестирование и отладка 3D-игр"]
    }
}



github_token = 'ваш токен'