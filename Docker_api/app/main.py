from typing import Union
from os import listdir
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
from process import categories

app = FastAPI(title="FastAPI news filtration and classification")
app.status = ""
input_file_name = "source.csv"

def printApp(msg):
    app.status += msg + "\n"

@app.get("/")
def read_root():
    return {"current_status": app.status,"Local machine contains:": listdir('.')}

@app.get("/calculate/", response_class=StreamingResponse)
async def export_data():
    df = categories(input_file_name, app) 
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        with open(input_file_name, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    printApp("Uploaded") 
    return {"message": f"Successfully uploaded {file.filename} as {input_file_name}"}