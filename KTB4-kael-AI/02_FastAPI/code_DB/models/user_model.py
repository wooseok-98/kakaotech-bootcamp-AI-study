from typing import Optional
from sqlmodel import Field, SQLModel, Session, select
from database import engine


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True)
    password: str
    nickname: str
    profile_image: Optional[str] = None


def get_all_users() -> list[dict]:
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return [u.model_dump() for u in users]


def get_users_by_id(user_id: int) -> Optional[dict]:
    with Session(engine) as session:
        user = session.get(User, user_id)
        return user.model_dump() if user else None


def get_users_by_email(email: str) -> Optional[dict]:
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        return user.model_dump() if user else None


def add_user(user: dict) -> dict:
    with Session(engine) as session:
        db_user = User(**user)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user.model_dump()
