from __future__ import annotations

import importlib
import io
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.extensions.logger import create_logger

logger = create_logger(__name__)

boto3_module: Any = None
botocore_config_module: Any = None
try:  # pragma: no cover
    boto3_module = importlib.import_module("boto3")
    botocore_config_module = importlib.import_module("botocore.client")
except Exception:
    boto3_module = None
    botocore_config_module = None


class R2StorageService:
    """Cloudflare R2 helper for uploading and retrieving Docling assets."""

    def __init__(self) -> None:
        self.enabled = bool(settings.R2_ENABLED)
        self.bucket = settings.R2_BUCKET
        self.account_id = settings.R2_ACCOUNT_ID
        self.endpoint_url = (
            settings.R2_ENDPOINT_URL
            or (
                f"https://{self.account_id}.r2.cloudflarestorage.com"
                if self.account_id
                else None
            )
        )
        self.access_key_id = settings.R2_ACCESS_KEY_ID
        self.secret_access_key = settings.R2_SECRET_ACCESS_KEY
        self.public_base_url = settings.R2_PUBLIC_BASE_URL
        self.asset_prefix = (settings.R2_ASSET_PREFIX or "docling-assets").strip("/")
        self.pdf_prefix = (settings.R2_PDF_PREFIX or "source-pdfs").strip("/")
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(
            self.enabled
            and self.bucket
            and self.endpoint_url
            and self.access_key_id
            and self.secret_access_key
            and boto3_module is not None
            and botocore_config_module is not None
        )

    def _get_client(self):
        if not self.is_configured:
            return None

        if self._client is None:
            config_cls = getattr(botocore_config_module, "Config", None)
            if config_cls is None:
                return None

            self._client = boto3_module.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=config_cls(signature_version="s3v4"),
                region_name="auto",
            )
        return self._client

    def get_public_url(self, key: str) -> Optional[str]:
        if not key:
            return None

        normalized = key.lstrip("/")
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{normalized}"

        if self.endpoint_url and self.bucket:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{normalized}"

        return None

    def build_asset_key(self, paper_id: str, relative_path: str) -> str:
        rel = relative_path.replace("\\", "/").lstrip("/")
        if self.asset_prefix:
            return f"{self.asset_prefix}/{paper_id}/{rel}"
        return f"{paper_id}/{rel}"

    def build_pdf_key(self, paper_id: str, filename: str = "source.pdf") -> str:
        normalized_name = (filename or "source.pdf").replace("\\", "/").split("/")[-1]
        if self.pdf_prefix:
            return f"{self.pdf_prefix}/{paper_id}/{normalized_name}"
        return f"{paper_id}/{normalized_name}"

    def upload_file(self, local_path: str | Path, key: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None or not self.bucket:
            return None

        path = Path(local_path)
        if not path.exists() or not path.is_file():
            return None

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        try:
            with path.open("rb") as file_obj:
                client.upload_fileobj(
                    file_obj,
                    self.bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                )

            return {
                "key": key,
                "url": self.get_public_url(key),
                "content_type": content_type,
                "size_bytes": path.stat().st_size,
            }
        except Exception as error:
            logger.warning(f"Failed to upload {path} to R2 key {key}: {error}")
            return None

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None or not self.bucket:
            return None

        try:
            stream = io.BytesIO(data)
            client.upload_fileobj(
                stream,
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            return {
                "key": key,
                "url": self.get_public_url(key),
                "content_type": content_type,
                "size_bytes": len(data),
            }
        except Exception as error:
            logger.warning(f"Failed to upload bytes to R2 key {key}: {error}")
            return None

    def get_object_bytes(self, key: str) -> Optional[bytes]:
        client = self._get_client()
        if client is None or not self.bucket:
            return None

        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
            body = response.get("Body")
            return body.read() if body else None
        except Exception as error:
            logger.warning(f"Failed to read R2 object {key}: {error}")
            return None

    def get_presigned_download_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        client = self._get_client()
        if client is None or not self.bucket:
            return None

        try:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as error:
            logger.warning(f"Failed to create pre-signed URL for {key}: {error}")
            return None

    def list_keys(self, prefix: str, max_keys: int = 200) -> List[str]:
        client = self._get_client()
        if client is None or not self.bucket:
            return []

        try:
            response = client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            rows = response.get("Contents", []) or []
            return [str(row.get("Key")) for row in rows if row.get("Key")]
        except Exception as error:
            logger.warning(f"Failed to list R2 keys under {prefix}: {error}")
            return []
