import unittest

from app.import_process.agent.markdown_image_rewriter import replace_markdown_images


class MarkdownImageRewriterTest(unittest.TestCase):
    def test_preserves_backslashes_in_image_summary(self):
        md_content = "![](images/formula.png)"
        image_infos = {
            "formula.png": (
                r"图片内容为 $-\sqrt{y}$",
                "http://127.0.0.1:9000/bucket/formula.png",
            )
        }

        result = replace_markdown_images(md_content, image_infos)

        self.assertEqual(
            result,
            r"![图片内容为 $-\sqrt{y}$](http://127.0.0.1:9000/bucket/formula.png)",
        )

    def test_matches_image_file_names_with_regex_metacharacters(self):
        md_content = "![](images/a+b(1).png)"
        image_infos = {
            "a+b(1).png": (
                "图片说明",
                "http://127.0.0.1:9000/bucket/a+b(1).png",
            )
        }

        result = replace_markdown_images(md_content, image_infos)

        self.assertEqual(
            result,
            "![图片说明](http://127.0.0.1:9000/bucket/a+b(1).png)",
        )

    def test_leaves_unrelated_image_references_unchanged(self):
        md_content = "![](images/keep.png)\n![](images/replace.png)"
        image_infos = {
            "replace.png": (
                "新说明",
                "http://127.0.0.1:9000/bucket/replace.png",
            )
        }

        result = replace_markdown_images(md_content, image_infos)

        self.assertEqual(
            result,
            "![](images/keep.png)\n![新说明](http://127.0.0.1:9000/bucket/replace.png)",
        )


if __name__ == "__main__":
    unittest.main()
