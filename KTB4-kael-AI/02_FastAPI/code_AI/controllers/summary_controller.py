from fastapi import HTTPException
from models import post_model, comment_model
from ai import llm


def summarize_post(post_id: int):
    post = post_model.get_post_by_id(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")

    prompt = f"다음 게시글을 한국어로 3문장 이내로 요약해줘.\n\n제목: {post['title']}\n내용: {post['content']}"
    summary = llm.ask(prompt)
    return {"message": "게시글 요약 성공", "summary": summary}


def summarize_comments(post_id: int):
    if not post_model.get_post_by_id(post_id):
        raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")

    comments = comment_model.get_comments_by_post_id(post_id)
    if not comments:
        raise HTTPException(status_code=404, detail="댓글이 없습니다.")

    comments_text = "\n".join(f"- {c['content']}" for c in comments)
    prompt = f"다음 댓글들을 한국어로 3문장 이내로 요약해줘.\n\n{comments_text}"
    summary = llm.ask(prompt)
    return {"message": "댓글 요약 성공", "summary": summary}
