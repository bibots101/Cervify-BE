from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request,BackgroundTasks
from fastapi.responses import JSONResponse,FileResponse
import os
import uvicorn
import sys
from pipeline.run_pipeline import run_pipeline
from crud import create_user, check_user, create_image, create_prediction, delete_image, delete_user
from database import SessionLocal, User, Image, Prediction
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils.progress_tracker import clear_progress,get_progress
import asyncio
import traceback
from utils.image_encryption import load_key,save_encrypted_image
from cryptography.fernet import Fernet
import tempfile
import charset_normalizer
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("data/images/",exist_ok=True)
os.makedirs("data/progress/",exist_ok=True)
app.mount("/images", StaticFiles(directory="data/images"), name="images")
app.mount("/progress", StaticFiles(directory="data/progress"), name="progress")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("Unhandled error:", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {str(exc)}"})


def delete_file(path: str):
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting temp file: {e}")

@app.get("/get_image/{filename}")
def view_image(filename: str, username:str,background_tasks: BackgroundTasks):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            db.close()
            raise HTTPException(status_code=404, detail="User not found.")
        db.close()
        key = load_key()
        fernet = Fernet(key.encode())
        encrypted_path = os.path.join("data/images", filename)

        if not os.path.exists(encrypted_path):
            raise HTTPException(status_code=404, detail="Image Not found")

        with open(encrypted_path, "rb") as f:
            decrypted = fernet.decrypt(f.read())

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(decrypted)
        temp.close()

        background_tasks.add_task(delete_file, temp.name)

        return FileResponse(temp.name, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loading image failed: {str(e)}")

@app.post("/signup/")
async def signup(username: str = Form(...), password: str = Form(...), full_name: str = Form(...)):
    try:
        if check_user(username, password):
            raise HTTPException(status_code=400, detail="User already exists.")
        create_user(username, password, full_name)
        return {"message": "User created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        db.close()
        if not user or not check_user(username, password):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        return {
            "message": "Login successful.",
            "username": user.username,
            "full_name": user.full_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), username: str = Form(...), type: str = Form(...)):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            db.close()
            raise HTTPException(status_code=404, detail="User not found.")
        user_id = user.id
        db.close()

        image_bytes = await file.read()
        image_name = os.path.splitext(file.filename)[0]
        image_base_name =  os.path.basename(file.filename)

        encrypted_filename = f"{image_name}.enc"
        image_path = f"data/images/{encrypted_filename}"

        save_encrypted_image(image_bytes, encrypted_filename)
        
        clear_progress(image_name)
        if not os.path.exists("tmp"):
            os.makedirs("tmp",exist_ok=True)
        with open(f"tmp/{file.filename}", "wb") as temp_img:
            temp_img.write(image_bytes)
        
        final_df = await asyncio.to_thread(run_pipeline, f"tmp/{file.filename}",type)

        if final_df is None or final_df.empty:
            return JSONResponse(
                status_code=400,
                content={"detail": "Segmentation failed or no processable cells found."}
            )

        prediction_list = final_df.to_dict(orient="records")
        for prediction in prediction_list:
            prediction["image_path"] = image_path
        saved_image = create_image(user_id=user_id, image_path=image_path)
        for row in final_df.itertuples():
            create_prediction(
                image_id=saved_image.id,
                x1=row.x1,
                y1=row.y1,
                x2=row.x2,
                y2=row.y2,
                label=row.label,
                confidence=str(row.confidence)
            )
        if os.path.exists(f"tmp/{file.filename}"):
            os.remove(f"tmp/{file.filename}")
        return JSONResponse(content={"prediction": prediction_list})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/progress/{image_name}")
def get_progress_status(image_name: str):
    try:
        return get_progress(image_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progress fetch failed: {str(e)}")


@app.get("/history/")
async def get_history(username: str):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            db.close()
            raise HTTPException(status_code=404, detail="User not found.")

        images = db.query(Image).filter(Image.user_id == user.id).all()
        history = []
        for image in images:
            predictions = db.query(Prediction).filter(Prediction.image_id == image.id).all()
            for pred in predictions:
                history.append({
                    "image_path": image.image_path,
                    "timestamp": image.timestamp,
                    "x1": pred.x1,
                    "y1": pred.y1,
                    "x2": pred.x2,
                    "y2": pred.y2,
                    "label": pred.label,
                    "confidence": pred.confidence
                })
        db.close()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetching history failed: {str(e)}")
    

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.delete("/deleteImage/")
async def delete_image_route(image_name: str = Form(...), username: str = Form(...)):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            db.close()
            raise HTTPException(status_code=404, detail="User not found")

        image_path = f"data/images/{image_name}"
        image = db.query(Image).filter(Image.user_id == user.id, Image.image_path == image_path).first()
        if not image:
            db.close()
            raise HTTPException(status_code=404, detail="Image not found in history")
        db.close()
        delete_image(image)
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"message": "Image and associated predictions deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.delete("/delete_account/")
async def delete_account_route(username: str = Form(...), password: str = Form(...)):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            db.close()
            raise HTTPException(status_code=404, detail="User not found")
        if not check_user(username, password):
            db.close()
            raise HTTPException(status_code=401, detail="Invalid credentials")
        db.close()
        delete_user(user.id)
        return {"message": "User account and all data deleted successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Account deletion failed: {str(e)}")


if __name__ == "__main__":
    reload = True
    if getattr(sys, 'frozen', False):
        reload = False
    if reload:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
