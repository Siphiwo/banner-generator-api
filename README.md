# Banner Builder API

Backend service for an AI-powered SaaS that performs content-aware banner resizing.

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (async-first)
- **Server**: Uvicorn

## Local Development

Create and activate a virtual environment, then install dependencies using `pip`:

```bash
pip install -e ".[dev]"
```

Run the development server:

```bash
uvicorn app.main:app --reload
```

