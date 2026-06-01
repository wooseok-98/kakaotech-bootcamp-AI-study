from fastapi import HTTPException
from models import like_model, post_model

def get_likes(post_id: int, user_id: int = None):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    likes = like_model.get_likes_by_post_id(post_id)
    is_liked = like_model.get_like(post_id, user_id) is not None if user_id else False
    return {"message": "좋아요 조회 성공", "count": len(likes), "is_liked": is_liked}

def add_like(post_id: int, data: dict):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    if like_model.get_like(post_id, data["user_id"]):
        raise HTTPException(status_code=409, detail="이미 좋아요를 눌렀습니다.")

    new_like = {"post_id": post_id, "user_id": data["user_id"]}
    like_model.add_like(new_like)
    return {"message": "좋아요 성공"}

def delete_like(post_id: int, data: dict):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    like = like_model.delete_like(post_id, data["user_id"])
    if not like:
        raise HTTPException(status_code=404, detail="좋아요를 누르지 않은 게시글입니다.")
    return {"message": "좋아요 취소 성공"}