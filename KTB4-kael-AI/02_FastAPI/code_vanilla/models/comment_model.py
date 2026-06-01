_comments = []
_next_id = 1

def get_comments_by_post_id(post_id: int):
    return [c for c in _comments if c["post_id"] == post_id]

def get_comment_by_id(comment_id: int):
    for c in _comments:
        if c["id"] == comment_id:
            return c
    return None

def add_comment(comment: dict):
    global _next_id
    comment["id"] = _next_id
    _next_id += 1
    _comments.append(comment)
    return comment

def delete_comment(comment_id: int):
    comment = get_comment_by_id(comment_id)
    if comment:
        _comments.remove(comment)
    return comment