"""Build the independent five-input recording lab from its dedicated template."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
(root / 'src/lab-516.html').write_text(
    (root / 'src/templates/lab-516.html').read_text(encoding='utf-8'), encoding='utf-8')
