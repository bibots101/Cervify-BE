# Cervify-BE — Backend for CV Analyzer

A production-ready FastAPI backend that powers the Cervify Computer-Vision pipeline.

## Table of Contents
- Project Overview
- Features
- Quickstart
- Installation
- Configuration
- Running the server
- API Reference
- Project layout
- Security & Notes
- Troubleshooting
- Contributing
- License

## Project Overview

This repository contains the backend service for the Cervify CV pipeline. It exposes a FastAPI application that handles user management, encrypted image storage, pipeline invocation, and result persistence.

Key responsibilities:
- Accept image uploads and run the analysis pipeline.
- Persist users, images and predictions in a local SQLite database.
- Serve encrypted images and progress files.

## Features
- FastAPI application with endpoints for signup, login, predict, history, progress and health checks.
- Image encryption before saving to disk.
- SQLAlchemy models for `User`, `Image` and `Prediction` stored in `database.db`.
- Integration with the repo's CV pipeline (YOLO/DINO/SVM/MLP models).

## Quickstart
Requirements: Python 3.10+ (project uses modern ML packages). Recommended to use a virtual environment.

1. Install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run the API (development):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Alternatively run the provided helper script `run_backend.bat` on Windows.

## Installation
- Clone the repo and create a virtualenv.
- Install Python dependencies using `requirements.txt`.
- Ensure model files (YOLO, Dino, classifiers) exist under `models/` (see `utils/global_var.py`).

Files to check:
- [main.py](main.py#L1)
- [database.py](database.py#L1)
- [requirements.txt](requirements.txt#L1)
- [utils/global_var.py](utils/global_var.py#L1)

## Configuration
- Database: Uses SQLite at `database.db` by default (see [database.py](database.py#L1)).
- Encryption key: `utils/global_var.py` contains `CRYPT_KEY` used for Fernet encryption. For production, replace this with an environment-driven secret and do NOT keep keys in source.
- Models: Put model files under `models/` to match paths in `utils/global_var.py`.

Recommended environment overrides (example):

```bash
# set a secure key in environment and update code to read it
export CERVIFY_CRYPT_KEY="<your-fernet-key>"
```

## Running the server
- Start with `uvicorn main:app` (development) or configure a production ASGI server.
- When run, the server creates `data/images/` and `data/progress/` folders and serves them as static mounts at `/images` and `/progress`.

## API Reference

All endpoints are JSON-based (multipart form for file upload). Below are the main routes and expected inputs.

- `GET /health` — basic health check.

- `POST /signup/` — create user (form-data)
  - `username` (form), `password` (form), `full_name` (form)

- `POST /login/` — login (form-data)
  - `username` (form), `password` (form)

- `POST /predict/` — run the pipeline (multipart/form-data)
  - `file` (UploadFile), `username` (form), `type` (form) — `type` controls the pipeline mode used by `pipeline.run_pipeline`.
  - Returns JSON with `prediction` list of bounding boxes and labels.
  - Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \\
  -F "file=@/path/to/image.jpg" \\
  -F "username=alice" \\
  -F "type=your_pipeline_mode"
```

- `GET /progress/{image_name}` — returns progress status for pipeline processing.

- `GET /get_image/{filename}` — serves a decrypted temporary image (query param `username` required).

- `GET /history/?username={username}` — returns saved prediction history for the given user.

- `DELETE /deleteImage/` — delete an image and associated predictions (form-data: `image_name`, `username`).

- `DELETE /delete_account/` — delete user account and all data (form-data: `username`, `password`).

For full behaviour, refer to the implementation in [main.py](main.py#L1).

## Project layout

- [main.py](main.py#L1) — FastAPI app and endpoints
- [database.py](database.py#L1) — SQLAlchemy models and DB setup
- [crud.py](crud.py#L1) — helper functions to create/read/delete DB records
- [pipeline/](pipeline/) — image processing and ML pipeline stages
- [utils/](utils/) — helpers for image encryption, loading, progress tracking

## Security & Notes
- The project currently includes a hardcoded `CRYPT_KEY` in `utils/global_var.py`. Replace this with a secure, environment-provided key before deploying.
- Store heavy ML model files outside the repository or in a protected storage; only pointers are kept in `utils/global_var.py`.

## Troubleshooting
- If prediction fails with segmentation errors, check the pipeline logs and `data/progress/` files.
- Ensure required model files exist and are compatible with the versions in `requirements.txt`.
- If the server cannot bind port 8000, choose another port or stop conflicting processes.

## Contributing
- Open issues and PRs. Keep changes focused and include tests where relevant.
- Respect the existing project style and dependency constraints.

## License
See the repository LICENSE file.

---

If you'd like, I can:
- Add example Postman/HTTPie requests for every endpoint.
- Add a docker-compose or Dockerfile for containerized deployment.
- Convert the hardcoded `CRYPT_KEY` to an env-var-backed loader and update code accordingly.

Please tell me which additions you'd like next.
