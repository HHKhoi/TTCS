def save_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)