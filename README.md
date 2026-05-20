# scigraphchat-dev

In the current containerized setup, the frontend runs as a Vite development server on port `3000`, and the backend runs as FastAPI on port `8000`. Frontend `/api/*` requests are proxied by Vite to the backend service inside Docker.

## Project Layout

```text
.
|-- implementation/
|   |-- backend/
|   |-- frontend/
|   |-- docker-compose.yml
|-- README.md
```

## Recommended: Docker Startup

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Internet access for first image build (Python deps, Guardrails validators, model downloads)

### 1. Create env files

From `implementation`:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
```

Fill at least these in `backend/.env`:

```env
ORKG_EMAIL=
ORKG_PASSWORD=
OPENAI_API_KEY=
VDL_API_KEY=
```

Do not commit `backend/.env`.

### 2. First-time Guardrails token setup

The backend Docker build installs Guardrails Hub validators. This requires `GUARDRAILS_TOKEN` during image build.

Copy the env example at the root of `implementation/` and fill in your token:

```powershell
Copy-Item implementation/.env.example implementation/.env
```

Open `implementation/.env` and set:

```env
GUARDRAILS_TOKEN=<your token>
```

Get or rotate the token at `https://hub.guardrailsai.com/keys`. Docker Compose reads `implementation/.env` automatically — no shell variable needed.

### 3. Start the stack

```powershell
cd implementation
docker compose up --build
```

### 4. Open the app

- Frontend: `http://localhost:3000`

## Runtime Architecture

```text
Browser
  -> Frontend Vite container (public, :3000)
     -> serves the frontend in dev mode
     -> proxies /api/* to backend
  -> Backend FastAPI container (public, :8000)
```

## Local Frontend Development With Vite

The Docker setup already uses Vite. If you want to run only the frontend outside Docker:

```powershell
cd implementation/frontend
npm install
npm run dev
```

- `npm run dev` starts the Vite development server with hot reload.

## What to do after the first successful build

For normal restarts:

```powershell
cd implementation
docker compose up
```

`GUARDRAILS_TOKEN` is needed again only when the backend image is rebuilt, for example after:

- `docker compose build backend`
- `docker compose up --build`
- a full cache/prune-driven rebuild

You can clear the shell variable after startup:

```powershell
Remove-Item Env:\GUARDRAILS_TOKEN
```

## Stop services

```powershell
cd implementation
docker compose down
```

## Windows: Enable Long File Path Support

> **Heads up for Windows users!** This repo contains deeply nested file paths that exceed Windows' default 260-character limit. Git will fail with `Filename too long` errors unless you enable long path support.

**1. Enable long paths in Git**:

```powershell
git config core.longpaths true
```

Or globally:

```powershell
git config --global core.longpaths true
```

**2. Enable long paths in Windows** (admin PowerShell):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

After both steps, retry your git operation.

## Optional: Manual local development

If you need non-Docker development, run the backend and frontend separately from `implementation/backend` and `implementation/frontend`. The Docker path above is the maintained setup.