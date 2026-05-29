from fastapi import APIRouter
from controllers import post_controller

router = APIRouter(prefix="/posts")

@router.get("/")
def get_posts():
    return post_controller.get_posts()

@router.post("/")
def create_post(post: dict):
    return post_controller.create_post(post)

@router.get("/{post_id}")
def get_post(post_id: int):
    return post_controller.get_post(post_id)

@router.delete("/{post_id}")
def delete_post(post_id: int):
    return post_controller.delete_post(post_id)