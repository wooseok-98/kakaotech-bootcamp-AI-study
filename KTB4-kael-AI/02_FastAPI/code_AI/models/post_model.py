from typing import Optional
from sqlmodel import Field, SQLModel, Session, select
from database import engine


class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    image: Optional[str] = None
    views: int = 0
    user_id: int


def get_all_posts() -> list[dict]:
    with Session(engine) as session:
        posts = session.exec(select(Post)).all()
        return [p.model_dump() for p in posts]


def get_post_by_id(post_id: int) -> Optional[dict]:
    with Session(engine) as session:
        post = session.get(Post, post_id)
        return post.model_dump() if post else None


def add_post(post: dict) -> dict:
    with Session(engine) as session:
        db_post = Post(**post)
        session.add(db_post)
        session.commit()
        session.refresh(db_post)
        return db_post.model_dump()


def update_post(post_id: int, data: dict) -> Optional[dict]:
    with Session(engine) as session:
        post = session.get(Post, post_id)
        if not post:
            return None
        for key, value in data.items():
            setattr(post, key, value)
        session.add(post)
        session.commit()
        session.refresh(post)
        return post.model_dump()


def delete_post(post_id: int) -> Optional[dict]:
    with Session(engine) as session:
        post = session.get(Post, post_id)
        if not post:
            return None
        post_dict = post.model_dump()
        session.delete(post)
        session.commit()
        return post_dict
