from cryptography.fernet import Fernet
import os
from utils.global_var import CRYPT_KEY

def load_key():
    return CRYPT_KEY

def save_encrypted_image(image_bytes, filename):
    key = load_key()
    fernet = Fernet(key.encode())
    encrypted = fernet.encrypt(image_bytes)

    save_path = os.path.join("data/images", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(encrypted)
