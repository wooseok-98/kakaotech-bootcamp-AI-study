from fastapi import APIRouter
from controllers import like_controller

router = APIRouter(prefix="/posts")

@router.get("/{post_id}/likes")
def get_likes(post_id: int, user_id: int = None):
    return like_controller.get_likes(post_id, user_id)

@router.post("/{post_id}/likes")
def add_like(post_id: int, data: dict):
    return like_controller.add_like(post_id, data)

@router.delete("/{post_id}/likes")
def delete_like(post_id: int, user_id: int):
    return like_controller.delete_like(post_id, {"user_id": user_id})