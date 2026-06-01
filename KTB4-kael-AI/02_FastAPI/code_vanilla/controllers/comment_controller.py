from fastapi import HTTPException
from models import comment_model, post_model

def get_comments(post_id: int):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    comments = comment_model.get_comments_by_post_id(post_id)
    return {"message": "댓글 목록 조회 성공", "data": comments}

def create_comment(post_id: int, data: dict):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    if not data.get("content"):
        raise HTTPException(status_code=400, detail="댓글 내용을 입력해주세요.")

    new_comment = {
        "content": data["content"],
        "post_id": post_id,
        "user_id": data["user_id"]
    }

    comment_model.add_comment(new_comment)
    return {"message": "댓글 작성 성공", "data": new_comment}

def delete_comment(post_id: int, comment_id: int):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
    comment = comment_model.delete_comment(comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="존재하지 않는 댓글입니다.")
    return {"message": "댓글 삭제 성공"}