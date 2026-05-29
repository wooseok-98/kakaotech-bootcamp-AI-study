_likes = []

def get_likes_by_post_id(post_id: int):
    return [l for l in _likes if l["post_id"] == post_id]

def get_like(post_id: int, user_id: int):
    for l in _likes:
        if l["post_id"] == post_id and l["user_id"] == user_id:
            return l
    return None

def add_like(like: dict):
    _likes.append(like)
    return like

def delete_like(post_id: int, user_id: int):
    like = get_like(post_id, user_id)
    if like:
        _likes.remove(like)
    return like