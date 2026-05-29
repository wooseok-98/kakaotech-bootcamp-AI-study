from typing import Optional
from sqlmodel import Field, SQLModel, Session, select
from database import engine


class Comment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    post_id: int
    user_id: int


def get_comments_by_post_id(post_id: int) -> list[dict]:
    with Session(engine) as session:
        comments = session.exec(select(Comment).where(Comment.post_id == post_id)).all()
        return [c.model_dump() for c in comments]


def get_comment_by_id(comment_id: int) -> Optional[dict]:
    with Session(engine) as session:
        comment = session.get(Comment, comment_id)
        return comment.model_dump() if comment else None


def add_comment(comment: dict) -> dict:
    with Session(engine) as session:
        db_comment = Comment(**comment)
        session.add(db_comment)
        session.commit()
        session.refresh(db_comment)
        return db_comment.model_dump()


def delete_comment(comment_id: int) -> Optional[dict]:
    with Session(engine) as session:
        comment = session.get(Comment, comment_id)
        if not comment:
            return None
        comment_dict = comment.model_dump()
        session.delete(comment)
        session.commit()
        return comment_dict
