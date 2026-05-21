import os
import argparse

# stuey/notes 디렉토리 설정
NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "study", "notes")

# 노트 폴더에서 .md 파일 목록을 가져와 정렬
def get_notes():
    files = [f for f in os.listdir(NOTES_DIR) if f.endswith(".md")]
    return sorted(files)

# 노트 목록과 각 노트의 제목(첫 줄)을 출력
def cmd_list():
    notes = get_notes()
    if not notes:
        print("노트가 없습니다.")
        return

    print("=== 노트 목록 ===\n")
    for i, note in enumerate(notes, 1):
        path = os.path.join(NOTES_DIR, note)
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip().lstrip("#").strip()
        print(f"{i}. {note}  —  {first_line}")


# 모든 노트에서 키워드가 포함된 줄을 찾아 파일명과 줄 번호와 함께 출력
def cmd_search(keyword):
    notes = get_notes()
    results = []

    for note in notes:
        path = os.path.join(NOTES_DIR, note)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        # 대소문자 구분 없이 검색, 빈 줄은 제외
        matches = [
            (i + 1, line.strip())
            for i, line in enumerate(lines)
            if keyword.lower() in line.lower() and line.strip()
        ]
        if matches:
            results.append((note, matches))

    if not results:
        print(f"'{keyword}'에 대한 검색 결과가 없습니다.")
        return

    print(f"=== '{keyword}' 검색 결과 ===\n")
    for note, matches in results:
        print(f"[{note}]")
        for lineno, line in matches:
            print(f"  {lineno}번째 줄: {line}")
        print()


# 특정 노트 파일의 전체 내용을 출력
def cmd_show(filename):
    path = os.path.join(NOTES_DIR, filename)
    if not os.path.exists(path):
        print(f"'{filename}' 파일을 찾을 수 없습니다.")
        return

    with open(path, encoding="utf-8") as f:
        print(f.read())


# 터미널 명령어(list, search, show)를 파싱해서 해당 함수 실행
def main():
    parser = argparse.ArgumentParser(description="학습 노트 검색 CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="노트 목록 보기")

    search_parser = subparsers.add_parser("search", help="키워드 검색")
    search_parser.add_argument("keyword", help="검색할 키워드")

    show_parser = subparsers.add_parser("show", help="노트 내용 보기")
    show_parser.add_argument("filename", help="파일명")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "search":
        cmd_search(args.keyword)
    elif args.command == "show":
        cmd_show(args.filename)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
