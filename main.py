from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()


@app.post("/cifar-classifier/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
