import html
import re
import unicodedata
from itertools import product
from pathlib import Path

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

ENGLISH_SPLIT_RE = re.compile(
    r"(?<=[A-Za-z])\s+(n['’]t|['’](?:s|re|ve|ll|d|m|t))\b|\b(gon|wan|got)\s+(na|ta)\b",
    re.IGNORECASE,
)

VIETNAMESE_GLUED_RE = re.compile(
    r"(?<=[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ])"
    r"(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ])"
)

# Chuẩn hóa HTML, Unicode, khoảng trắng
def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

# Chuản hóa tiếng anh
def normalize_english_text(text: str) -> str:
    text = normalize_text(text)
    return ENGLISH_SPLIT_RE.sub(lambda m: "".join(g for g in m.groups() if g), text)

# Chuẩn hóa tiếng việt
def normalize_vietnamese_text(text: str) -> str:
    text = normalize_text(text)
    return VIETNAMESE_GLUED_RE.sub(" ", text)


def normalize_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    if input_path.name.endswith(".en.txt"):
        normalizer = normalize_english_text
    elif input_path.name.endswith(".vi.txt"):
        normalizer = normalize_vietnamese_text
    else:
        normalizer = normalize_text
    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("x", encoding="utf-8", newline="\n") as output_file,
    ):
        for total_lines, line in enumerate(input_file, start=1):
            output_file.write(normalizer(line) + "\n")

    print(f"[OK] {output_path}: {total_lines} lines")

def main() -> None:
    for corpus, split, language in product(
        ("IWSLT", "PhoMT"), ("train", "dev", "test"), ("en", "vi")
    ):
        relative_path = Path(corpus) / split / f"{split}.{language}.txt"
        normalize_file(INPUT_DIR / relative_path, OUTPUT_DIR / relative_path)

    print(f"Done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
