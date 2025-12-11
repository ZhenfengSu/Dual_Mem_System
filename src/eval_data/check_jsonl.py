import json
# 检查jsonl是否能正常加载，如果无法正常加载，则去除多余的空行
import sys
import os
def check_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        print(f"Successfully loaded {len(items)} items from {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading {file_path}: {e}")
        sys.exit(1)
    return items
if __name__ == "__main__":
    file_path = './gaia.jsonl'  # 替换为你的jsonl文件路径
    check_jsonl(file_path)