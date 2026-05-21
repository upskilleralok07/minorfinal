# minorfinal

## Deployment

Quick steps to deploy and run the app:

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set the Groq API key (if you use Groq features):

```powershell
$env:GROQ_API_KEY = "<your_groq_api_key>"
```

4. Run the Streamlit app:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Notes:
- Keep API keys out of the repository; use environment variables or your deployment platform's secret manager.
- Consider removing large model artifacts from the repo and storing them in releases or a storage bucket; `.gitignore` already excludes `*.pkl` and `placepilot.db`.

