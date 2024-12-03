Situated/                   # Root directory for your project
│
├── manage.py               # Django management script
├── Situated/               # Project folder (contains settings and main config)
│   ├── __init__.py
│   ├── settings.py         # Project settings
│   ├── urls.py             # Project URL configuration
│   ├── asgi.py
│   └── wsgi.py
│
├── blrs/                   # App folder for BLRS
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py           # Django models
│   ├── tests.py
│   ├── views.py            # Django views for BLRS
│   ├── urls.py             # App-specific URLs (optional)
│   │
│   ├── templates/          # HTML templates
│   │   └── blrs/
│   │       └── blrs.html   # Main HTML file for BLRS
│   │
│   └── static/             # Static files directory
│       ├── css/
│       │   └── blrs.css    # CSS file for styling
│       └── js/
│           └── blrs.js     # JavaScript file for frontend logic
│
└── db.sqlite3              # Database file (if using SQLite)
