import re
from collections.abc import Mapping


def compile_markdown_image_reference_pattern(image_file: str) -> re.Pattern:
    escaped_image_file = re.escape(image_file)
    return re.compile(r"!\[[^\]]*]\([^)]*" + escaped_image_file + r"[^)]*\)")


def replace_markdown_images(
    md_content: str,
    image_infos: Mapping[str, tuple[str, str]],
) -> str:
    for image_file, (summary, url) in image_infos.items():
        pattern = compile_markdown_image_reference_pattern(image_file)
        md_content = pattern.sub(lambda _: f"![{summary}]({url})", md_content)
    return md_content
