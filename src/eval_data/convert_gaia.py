import json
import os

def convert_to_jsonl():
    input_path = './gaia.json'
    output_path = './gaia.jsonl'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用 JSONDecoder 逐个解析拼接在一起的 JSON 对象
    decoder = json.JSONDecoder()
    pos = 0
    items = []

    while pos < len(content):
        # 跳过空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        
        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            items.append(obj)
            pos = end_pos
        except json.JSONDecodeError as e:
            print(f"Parsing error at position {pos}: {e}")
            break

    print(f"Successfully parsed {len(items)} items.")

    # 写入 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted file saved to: {output_path}")

if __name__ == "__main__":
    convert_to_jsonl()