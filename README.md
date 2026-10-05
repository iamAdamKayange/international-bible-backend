# True International Bible Backend

Optional FastAPI backend for anonymous-install push notifications, Firebase Cloud Messaging, daily Bible verse notifications, reading reminders, localized announcements and remote configuration. It also provides Scripture Connections ranked from verified references in the bundled corpus. The service has no login or user accounts. Bible reading, search and user library data remain local in the Flutter app.

## Deploy on Render

This repository includes `render.yaml`. Connect the GitHub repository to Render and create the Blueprint. It provisions the Python API and a Postgres database. Render supplies `PORT`; the Blueprint wires `DATABASE_URL` automatically.

Set this secret in the Render service before enabling FCM delivery:

- `FIREBASE_SERVICE_ACCOUNT_JSON`: the Firebase service-account JSON for the Firebase project that owns the app's FCM tokens. Store it as a Render secret; do not commit it.

Optional public remote configuration:

- `REMOTE_CONFIG_JSON`: JSON containing public feature flags and localized announcements. The Blueprint sets an empty announcement list with Scripture Connections enabled by default.

Render sets `PYTHON_VERSION` from the Blueprint. No Flutter or Firebase client config is required in this backend repository. Once deployed, configure the Flutter app with `--dart-define=TIB_API_BASE_URL=https://<your-render-service>.onrender.com`.

## Local run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://..."
uvicorn app.main:app --reload
```

`GET /health` reports service readiness. See the API source for routes: `/v1/config`, `/v1/devices/register`, `/v1/scripture/related`, and the reserved future premium endpoints. Premium verification currently grants no entitlement.

## Scripture index

The checked-in `app/data` corpus and semantic model were generated from the project's existing Bible dataset. All returned verse references are looked up from that corpus; the service does not generate Bible text or references. To rebuild intentionally, install `requirements-build.txt` and run:

```powershell
python scripts/build_semantic_index.py --source <path-to-en_kjv.json> --output app/data
```

Review source data permissions before redistributing or rebuilding the corpus.
