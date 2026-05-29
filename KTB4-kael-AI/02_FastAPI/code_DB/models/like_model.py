from typing import Optional
from sqlmodel import Field, SQLModel, Session, select
from database import engine


class Like(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    post_id: int
    user_id: int


def get_likes_by_post_id(post_id: int) -> list[dict]:
    with Session(engine) as session:
        likes = session.exec(select(Like).where(Like.post_id == post_id)).all()
        return [l.model_dump() for l in likes]


def get_like(post_id: int, user_id: int) -> Optional[dict]:
    with Session(engine) as session:
        like = session.exec(
            select(Like).where(Like.post_id == post_id, Like.user_id == user_id)
        ).first()
        return like.model_dump() if like else None


def add_like(like: dict) -> dict:
    with Session(engine) as session:
        db_like = Like(**like)
        session.add(db_like)
        session.commit()
        session.refresh(db_like)
        return db_like.model_dump()


def delete_like(post_id: int, user_id: int) -> Optional[dict]:
    with Session(engine) as session:
        like = session.exec(
            select(Like).where(Like.post_id == post_id, Like.user_id == user_id)
        ).first()
        if not like:
            return None
        like_dict = like.model_dump()
        session.delete(like)
        session.commit()
        return like_dict
