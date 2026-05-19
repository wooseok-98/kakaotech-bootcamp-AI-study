# 인증 필수 x: S2S
# 게시글 작성
    - 제목
    - 내용
    - 이미지 x (파일 첨부)
    - 글 작성
        POST /posts
        ("title":"제목, "content":내용")
    - 응답 성공 200
        생성된 글의 번허
        {"post_id":3}

    - 글 목록
        GET /list: RPC 시스템 x
        GET /posts: REST API
    - 응답 성공 200
        {"id":3, "title":제목, "content": 내용, 좋아요 등}

