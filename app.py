import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import companyDataAnalysis

# load config
with open("config.json") as f:
    config = json.load(f)


app = FastAPI()
app.add_middleware(CORSMiddleware, 
                   allow_origins=config["CORS"]["origins"], 
                   allow_methods=config["CORS"]["methods"], 
                   allow_headers=config["CORS"]["headers"])


app.include_router(companyDataAnalysis.router)


if __name__ == "__main__":
    uvicorn.run("app:app",
                host=config["server"]["host"],
                port=config["server"]["port"], 
                reload=config["server"]["reload"], timeout_keep_alive=500)