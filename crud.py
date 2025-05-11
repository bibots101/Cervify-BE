from database import SessionLocal, User, Image, Prediction
import bcrypt
import os

def create_user(username: str, password: str, full_name: str):
    db = SessionLocal()
    try:
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = User(username=username, password_hash=hashed_pw.decode('utf-8'), full_name=full_name)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def delete_user(user_id: int):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return

        images = db.query(Image).filter(Image.user_id == user_id).all()
        for img in images:
            delete_image_by_id(img.id)
            if os.path.exists(img.image_path):
                os.remove(img.image_path)

        db.delete(user)
        db.commit()
    finally:
        db.close()

def check_user(username: str, password: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8'))
    finally:
        db.close()

def create_image(user_id: int, image_path: str):
    db = SessionLocal()
    try:
        image = Image(user_id=user_id, image_path=image_path)
        db.add(image)
        db.commit()
        db.refresh(image)
        return image
    finally:
        db.close()

def delete_image_by_id(image_id: int):
    db = SessionLocal()
    try:
        image = db.query(Image).filter(Image.id == image_id).first()
        if image:
            db.delete(image)
            if os.path.exists(image.image_path):
                os.remove(image.image_path)
            db.commit()
    finally:
        db.close()

def delete_image(image: Image):
    db = SessionLocal()
    try:
        db.query(Prediction).filter(Prediction.image_id == image.id).delete()
        db.delete(image)
        db.commit()
    finally:
        db.close()


def create_prediction(image_id: int, x1: float, y1: float, x2: float, y2: float, label: str, confidence: str):
    db = SessionLocal()
    try:
        prediction = Prediction(
            image_id=image_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=label,
            confidence=confidence
        )
        db.add(prediction)
        db.commit()
        return prediction
    finally:
        db.close()
