from fastapi import APIRouter
from controllers import summary_controller

router = APIRouter(prefix="/posts")


@router.get("/{post_id}/summary")
def summarize_post(post_id: int):
    return summary_controller.summarize_post(post_id)


@router.get("/{post_id}/comments/summary")
def summarize_comments(post_id: int):
    return summary_controller.summarize_comments(post_id)
