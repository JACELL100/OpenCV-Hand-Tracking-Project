# Hand Tracking (Django + OpenCV)

This repository contains a small Django project for hand tracking using OpenCV.

## Requirements

- Python 3.8+ (tested with 3.8-3.11)

## Install

Install the Python dependencies with pip:

```bash
pip install Django opencv-python numpy Pillow
```

(If you're using a virtual environment: create and activate it before running the install command.)

## Setup

1. Ensure you're in the project root (where `manage.py` is located):

```powershell
cd c:\Desktop\hand_tracking_project\hand_tracking
```

2. Apply migrations (the project includes a SQLite DB file `db.sqlite3` already, but running migrations is safe/required if you change models):

```powershell
python manage.py migrate
```

3. (Optional) Create a superuser to access the admin:

```powershell
python manage.py createsuperuser
```

## Run the development server

```powershell
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser to view the app.

## Notes

- The `tracker` app contains templates and views for hand tracking. The template `tracker/templates/index.html` is included in the project.
- If you plan to use your webcam with OpenCV, ensure no other application is using it.

## Troubleshooting

- If you get permission or camera access errors, try running with elevated privileges or check your OS camera privacy settings.
- For OpenCV video capture on Windows, if `opencv-python` has trouble opening the camera, try installing `opencv-contrib-python` instead.

## License

This project does not include a license file. Add one if you plan to publish the code.