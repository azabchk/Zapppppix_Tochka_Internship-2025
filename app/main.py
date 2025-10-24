from fastapi import FastAPI

app = FastAPI(title="zapppppix internship API")

@app.get("/health")
def health():
    return {"status": "ok"}
