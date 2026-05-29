from fastapi import HTTPException
from models import user_model

def create_user(data: dict):
    if user_model.get_users_by_email(data["email"]):
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다.")
    
    if data["password"] != data["password_confirm"]:
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
    
    new_user = {
        "email": data["email"],
        "password": data["password"],
        "nickname": data["nickname"],
        "profile_image": data.get("profile_image", None)
    }

    added_user = user_model.add_user(new_user)
    return {"message": "회원가입 성공", "user": added_user}

def login(data: dict):
    user = user_model.get_users_by_email(data["email"])
    if not user:
        raise HTTPException(status_code=404, detail="존재하지 않는 이메일입니다.")
    if user["password"] != data["password"]:
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")
    return {"message": "로그인 성공", "user": user}
