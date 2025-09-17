## AI Student Assistant (Prototype)

Production-oriented demo scaffold: Django + Postgres + Docker. Supports PDF/text upload, placeholder processing, and simple results page.

### Quickstart

1. Copy env

```bash
cp .env.example .env
```

2. Build & run

```bash
docker compose up --build
```

3. Migrate and create superuser (first run)

```bash
docker compose exec web python manage.py migrate
# optional
# docker compose exec web python manage.py createsuperuser
```

4. Open app

- Web: http://localhost:8000/
- Admin: http://localhost:8000/admin/

### Notes
- This is a demo pipeline; no external AI calls are made.
- Add your keys in `.env` and extend `assistant/utils/pipeline.py` for real LLMs.

