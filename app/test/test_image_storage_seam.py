import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.import_process.agent.image_storage import MinioImageStorage
from app.import_process.agent.nodes import node_md_img


class FakeImageStorage:
    def __init__(self):
        self.deleted_stems = []
        self.uploaded_images = []

    def replace_images_for_document(self, stem, image_targets):
        self.deleted_stems.append(stem)
        self.uploaded_images.extend(image_targets)
        return {
            image_file: f"http://storage.local/{stem}/{image_file}"
            for image_file, _ in image_targets
        }


class ImageStorageSeamTest(unittest.TestCase):
    def test_step_4_upload_uses_image_storage_interface(self):
        fake_storage = FakeImageStorage()
        md_content = "![](images/formula.png)"
        targets = [("formula.png", r"C:\tmp\formula.png", ("before", "after"))]
        summarise = {"formula.png": r"图片内容为 $-\sqrt{y}$"}

        with patch.object(node_md_img, "get_image_storage", return_value=fake_storage):
            result = node_md_img.step_4_upload_images_and_replace_md(
                summarise=summarise,
                targets=targets,
                md_content=md_content,
                stem="doc",
            )

        self.assertEqual(fake_storage.deleted_stems, ["doc"])
        self.assertEqual(fake_storage.uploaded_images, [("formula.png", r"C:\tmp\formula.png")])
        self.assertEqual(
            result,
            r"![图片内容为 $-\sqrt{y}$](http://storage.local/doc/formula.png)",
        )

    def test_uploaded_image_is_rewritten_when_summary_is_empty(self):
        fake_storage = FakeImageStorage()
        md_content = "![](images/course.png)"
        targets = [("course.png", r"C:\tmp\course.png", ("before", "after"))]

        with patch.object(node_md_img, "get_image_storage", return_value=fake_storage):
            result = node_md_img.step_4_upload_images_and_replace_md(
                summarise={},
                targets=targets,
                md_content=md_content,
                stem="doc",
            )

        self.assertEqual(
            result,
            "![](http://storage.local/doc/course.png)",
        )

    def test_minio_upload_failure_propagates(self):
        calls = []

        class FailingClient:
            def list_objects(self, **kwargs):
                calls.append(("list", kwargs["prefix"]))
                return [SimpleNamespace(object_name="upload-images/doc/old.png")]

            def remove_objects(self, *args, **kwargs):
                calls.append(("remove", None))
                return []

            def fput_object(self, **kwargs):
                calls.append(("upload", kwargs["object_name"]))
                raise RuntimeError("storage unavailable")

        with patch(
            "app.clients.minio_utils.get_minio_client",
            return_value=FailingClient(),
        ):
            with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
                MinioImageStorage().replace_images_for_document(
                    "doc",
                    [("course.png", r"C:\tmp\course.png")],
                )

        self.assertEqual(
            calls,
            [
                ("list", "upload-images/doc/"),
                ("upload", "upload-images/doc/course.png"),
            ],
        )

    def test_minio_uses_consistent_object_keys_and_real_content_types(self):
        calls = []

        class RecordingClient:
            def list_objects(self, **kwargs):
                calls.append(("list", kwargs["prefix"]))
                return [
                    SimpleNamespace(object_name="upload-images/doc/keep.png"),
                    SimpleNamespace(object_name="upload-images/doc/stale.jpg"),
                ]

            def fput_object(self, **kwargs):
                calls.append(
                    (
                        "upload",
                        kwargs["object_name"],
                        kwargs["content_type"],
                    )
                )

            def remove_objects(self, bucket_name, delete_objects):
                calls.append(
                    ("remove", [item._name for item in delete_objects])
                )
                return []

        class FakeDeleteObject:
            def __init__(self, name):
                self._name = name

        fake_minio_deleteobjects = SimpleNamespace(DeleteObject=FakeDeleteObject)
        fake_minio_utils = SimpleNamespace(get_minio_client=lambda: RecordingClient())

        with (
            patch.dict(
                "sys.modules",
                {
                    "minio.deleteobjects": fake_minio_deleteobjects,
                    "app.clients.minio_utils": fake_minio_utils,
                },
            ),
            patch("app.conf.minio_config.minio_config.minio_img_dir", "/upload-images"),
            patch("app.conf.minio_config.minio_config.minio_secure", False),
        ):
            urls = MinioImageStorage().replace_images_for_document(
                "doc",
                [
                    ("keep.png", r"C:\tmp\keep.png"),
                    ("note.webp", r"C:\tmp\note.webp"),
                ],
            )

        self.assertEqual(
            calls,
            [
                ("list", "upload-images/doc/"),
                ("upload", "upload-images/doc/keep.png", "image/png"),
                ("upload", "upload-images/doc/note.webp", "image/webp"),
                ("remove", ["upload-images/doc/stale.jpg"]),
            ],
        )
        self.assertTrue(urls["keep.png"].endswith("/upload-images/doc/keep.png"))


if __name__ == "__main__":
    unittest.main()
