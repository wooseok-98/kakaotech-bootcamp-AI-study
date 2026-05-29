from fastapi import APIRouter
from controllers import user_controller

router = APIRouter(prefix="/users")

@router.post("")
def create_user(data: dict):
    return user_controller.create_user(data)

@router.post("/login")
def login(data: dict):
    return user_controller.login(data)