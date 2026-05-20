# HubLink Frontend

React/Vite frontend for the HubLink ORKG question answering app.

## Current Setup

The frontend is development-only in Docker:

- Vite serves the app on `http://localhost:3000`
- Vite proxies `/api` requests to the backend via `VITE_DEV_PROXY_TARGET`
- Nginx is no longer used

The full stack now runs from the parent implementation folder with a single Compose file.

## Run with Docker

From `implementation`:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Frontend-Only Local Run

If you want to run only the frontend outside Docker:

```bash
npm install
npm run dev
```

By default, Vite proxies to `http://localhost:8000`. To point it elsewhere, set:

```bash
VITE_DEV_PROXY_TARGET=http://localhost:8000
```

## Features

- Chat interface for querying ORKG through HubLink retrieval approaches
- Knowledge graph visualization
- Configurable retrieval and model settings
- Backend health/status indicator

## WCAG (Web Content Accessibility Guidelines)

The web design follows the WCAG of WAI (Web Accessbility Initiative):
https://www.w3.org/WAI/standards-guidelines/wcag/
Current Version: https://www.w3.org/WAI/standards-guidelines/wcag/wcag3-intro/
