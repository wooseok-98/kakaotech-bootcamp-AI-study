_posts = []

def get_all_posts():
    return _posts

def get_post_by_id(post_id: int):
    for post in _posts:
        if post["id"] == post_id:
            return post
    return None

def add_post(post: dict):
    _posts.append(post)
    return post

def update_post(post_id: int, data: dict):
    post = get_post_by_id(post_id)
    if post:
        post.update(data)
    return post

def delete_post(post_id: int):
    post = get_post_by_id(post_id)
    if post:
        _posts.remove(post)
    return post