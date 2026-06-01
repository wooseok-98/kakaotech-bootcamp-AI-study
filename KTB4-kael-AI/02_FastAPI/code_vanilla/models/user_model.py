_users = []

def get_all_users():
    return _users  

def get_users_by_id(user_id: int):
    for u in _users:
        if u["id"] == user_id:
            return u
    return None

def get_users_by_email(user_email: str):
    for u in _users:
        if u["email"] == user_email:
            return u
    return None

def add_user(user: dict):
    _users.append(user)
    return user