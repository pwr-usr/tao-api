from fastapi import FastAPI
from tao import train_tao_model

app = FastAPI()


@app.get("/tao-service/api/v1/training")
def training():
    return train_tao_model()

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5460)