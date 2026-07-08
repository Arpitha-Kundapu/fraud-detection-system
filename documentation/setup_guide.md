# Environment Setup Guide

This guide walks through setting up your local Python development environment for the **End-to-End Fraud Detection System**.

## Prerequisites

Ensure you have the following installed on your machine:
- Python 3.8 or higher
- `pip` (Python package manager)
- Git (optional, for version control)

---

## Step 1: Create a Virtual Environment

It is highly recommended to isolate dependencies inside a virtual environment.

### Windows (cmd / PowerShell)
Open your terminal in the project root folder and execute:
```powershell
python -m venv .venv
```

### macOS / Linux
```bash
python3 -m venv .venv
```

---

## Step 2: Activate the Virtual Environment

### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
source .venv/bin/activate
```

*(You will know it is activated when you see `(.venv)` prepended to your command line prompt.)*

---

## Step 3: Install Requirements & Local Package

Install third-party packages, and then install the `src` directory as an editable local package. This enables absolute package-level importing.

```bash
# Install third-party packages
pip install -r requirements.txt

# Install the project library in editable development mode
pip install -e .
```

To confirm the local installation succeeded, run:
```bash
pip list | grep src
# Or on Windows PowerShell:
pip list | Select-String src
```
You should see `src` listed with a path pointing to your project directory.

---

## Step 4: Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and adjust the variables (such as `API_PORT` or `LOG_LEVEL`) to suit your local setup.

---

## Step 5: Start the API Server

Launch the development API server using `uvicorn`:
```bash
uvicorn api.app:app --reload
```
You should see log output indicating that the server is running on `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/health` or `http://127.0.0.1:8000/docs` to test endpoints.
