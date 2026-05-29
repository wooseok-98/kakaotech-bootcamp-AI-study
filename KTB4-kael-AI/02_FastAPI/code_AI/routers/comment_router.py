from fastapi import APIRouter
from controllers import comment_controller

router = APIRouter(prefix="/posts")

@router.get("/{post_id}/comments")
def get_comments(post_id: int):
    return comment_controller.get_comments(post_id)

@router.post("/{post_id}/comments")
def create_comment(post_id: int, data: dict):
    return comment_controller.create_comment(post_id, data)

@router.delete("/{post_id}/comments/{comment_id}")
def delete_comment(post_id: int, comment_id: int):
    return comment_controller.delete_comment(post_id, comment_id)