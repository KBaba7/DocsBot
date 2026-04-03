from pathlib import Path

from app.config import get_settings


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = self.settings.storage_backend.lower().strip()
        self._client = None
        self.settings.upload_path.mkdir(parents=True, exist_ok=True)

    def save_pdf(self, *, file_hash: str, filename: str, content: bytes) -> str:
        if self._use_supabase():
            key = self._build_supabase_key(file_hash=file_hash, filename=filename)
            self._supabase().storage.from_(self.settings.supabase_storage_bucket).upload(
                path=key,
                file=content,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )
            return f"supabase://{self.settings.supabase_storage_bucket}/{key}"

        suffix = Path(filename).suffix or ".pdf"
        target = self.settings.upload_path / f"{file_hash}{suffix}"
        if not target.exists():
            target.write_bytes(content)
        return str(target)

    def read_file_bytes(self, *, file_path: str) -> bytes:
        if file_path.startswith("supabase://"):
            bucket, key = self._parse_supabase_path(file_path)
            result = self._supabase().storage.from_(bucket).download(key)
            if isinstance(result, bytes):
                return result
            return bytes(result)
        return Path(file_path).read_bytes()

    def delete_file(self, *, file_path: str) -> None:
        if file_path.startswith("supabase://"):
            bucket, key = self._parse_supabase_path(file_path)
            try:
                self._supabase().storage.from_(bucket).remove([key])
            except Exception:
                pass
            return
        local_path = Path(file_path)
        if local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass

    def _use_supabase(self) -> bool:
        return (
            self.backend == "supabase"
            and bool(self.settings.supabase_url)
            and bool(self.settings.supabase_service_role_key)
        )

    def _supabase(self):
        if self._client is None:
            from supabase import create_client

            self._client = create_client(self.settings.supabase_url, self.settings.supabase_service_role_key)
        return self._client

    def _build_supabase_key(self, *, file_hash: str, filename: str) -> str:
        suffix = Path(filename).suffix or ".pdf"
        base_prefix = self.settings.supabase_storage_prefix.strip("/ ")
        if base_prefix:
            return f"{base_prefix}/{file_hash}{suffix}"
        return f"{file_hash}{suffix}"

    def _parse_supabase_path(self, file_path: str) -> tuple[str, str]:
        without_scheme = file_path.removeprefix("supabase://")
        parts = without_scheme.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid supabase path: {file_path}")
        return parts[0], parts[1]
