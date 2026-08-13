from collections.abc import Iterable
import mimetypes
from pathlib import Path
from typing import Protocol

from app.core.logger import logger


_IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


class ImageStorage(Protocol):
    def replace_images_for_document(
        self,
        stem: str,
        image_targets: Iterable[tuple[str, str]],
    ) -> dict[str, str]:
        """Replace document images in storage and return image file names mapped to public URLs."""


class MinioImageStorage:
    def replace_images_for_document(
        self,
        stem: str,
        image_targets: Iterable[tuple[str, str]],
    ) -> dict[str, str]:
        from minio.deleteobjects import DeleteObject

        from app.clients.minio_utils import get_minio_client
        from app.conf.minio_config import minio_config

        minio_client = get_minio_client()
        if minio_client is None:
            raise RuntimeError("MinIO 图片存储客户端不可用")
        image_targets = list(image_targets)
        object_prefix = f"{minio_config.minio_img_dir.strip('/')}/{stem}/"
        object_list = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=object_prefix,
            recursive=True,
        )
        old_object_names = {obj.object_name for obj in object_list}
        new_object_names = {
            f"{object_prefix}{image_file}" for image_file, _ in image_targets
        }

        image_urls = {}
        for image_file, image_path in image_targets:
            object_name = f"{object_prefix}{image_file}"
            try:
                minio_client.fput_object(
                    bucket_name=minio_config.bucket_name,
                    object_name=object_name,
                    file_path=image_path,
                    content_type=_IMAGE_CONTENT_TYPES.get(
                        Path(image_file).suffix.lower(),
                        mimetypes.guess_type(image_file)[0]
                        or "application/octet-stream",
                    ),
                )
                scheme = "https" if minio_config.minio_secure else "http"
                image_urls[image_file] = (
                    f"{scheme}://{minio_config.endpoint}/{minio_config.bucket_name}/"
                    f"{object_name}"
                )
            except Exception as e:
                logger.error(f"上传图片失败：{image_file},失败原因：{e}")
                raise

        stale_object_names = sorted(old_object_names - new_object_names)
        if stale_object_names:
            delete_object_list = [DeleteObject(name) for name in stale_object_names]
            errors = minio_client.remove_objects(
                minio_config.bucket_name,
                delete_object_list,
            )
            delete_errors = list(errors)
            if delete_errors:
                raise RuntimeError(f"删除旧图片失败：{delete_errors}")
        return image_urls


def get_image_storage() -> ImageStorage:
    return MinioImageStorage()
