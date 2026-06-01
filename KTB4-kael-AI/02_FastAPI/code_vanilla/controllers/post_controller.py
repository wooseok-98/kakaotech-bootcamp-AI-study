from fastapi import HTTPException
from models import post_model, like_model

def get_posts():
    posts = post_model.get_all_posts()
    result = []
    for post in posts:
        post_copy = dict(post)
        post_copy["like_count"] = len(like_model.get_likes_by_post_id(post["id"]))
        result.append(post_copy)
    return {"message": "게시글 목록 조회 성공", "data": result}

def create_post(data: dict):
    if not data.get("title") or not data.get("content"):
        raise HTTPException(status_code=400, detail="제목과 내용을 모두 입력해주세요.")
    
    new_post = {
        "id": len(post_model.get_all_posts()) + 1,
        "title": data["title"],
        "content": data["content"],
        "image": data.get("image", None),
        "views": 0,
        "user_id": data["user_id"]
        }

    post_model.add_post(new_post)
    return {"message": "게시글 작성 성공", "data": new_post}

def get_post(post_id: int):
    post = post_model.get_post_by_id(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    
    post_model.update_post(post_id, {"views": post["views"] + 1})
    return {"message": "게시글 조회 성공", "data": post}

def delete_post(post_id: int):
    post = post_model.delete_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    
    return {"message": "게시글 삭제 성공"}