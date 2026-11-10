"""Typed-by-convention client for published file objects."""

import hashlib
from pathlib import Path
from typing import Any

from services.common import Remote, identifier


class Catalog(Remote):
    """Storage client keeping local paths out of public resource identifiers."""

    def get(self, kind: str, key: str) -> dict[str, Any]:
        """Read an object manifest.

        Parameters:
            kind (str): Collection name.
            key (str): Object ID.

        Returns:
            dict[str, Any]: Published manifest.
        """
        return self.json("GET", f"/objects/{identifier(kind)}/{identifier(key)}")

    def reserve(
        self, kind: str, key: str, name: str, references: dict[str, str], token: str
    ) -> dict[str, Any]:
        """Create a job-owned result draft before its inputs can be consumed.

        Parameters:
            kind (str): Result collection.
            key (str): Job and result ID.
            name (str): Initial result name.
            references (dict[str, str]): Published inputs protected by this draft.
            token (str): Private controller capability, never exposed in manifests.

        Returns:
            dict[str, Any]: Reserved draft manifest.
        """
        return self.json(
            "POST",
            f"/objects/{identifier(kind)}",
            json={
                "id": key,
                "name": name,
                "references": references,
                "job_id": key,
                "reservation_token": token,
            },
        )

    def discard(self, kind: str, key: str, token: str | None = None) -> None:
        """Delete a failed job draft using its private reservation capability.

        Parameters:
            kind (str): Result collection.
            key (str): Result ID.
            token (str | None): Owner capability, absent for ordinary drafts.

        Returns:
            None: Releases protected input references if the draft exists.
        """
        self.request(
            "DELETE",
            f"/drafts/{identifier(kind)}/{identifier(key)}",
            headers={"X-Reservation-Token": token} if token else {},
        )

    def publish(
        self,
        kind: str,
        key: str,
        name: str,
        metadata: dict[str, Any],
        references: dict[str, str],
        files: dict[str, Path],
        reservation_token: str | None = None,
    ) -> dict[str, Any]:
        """Fill a reserved result or create an ordinary draft, then publish its files.

        Parameters:
            kind (str): Collection name.
            key (str): New object ID.
            name (str): Display name.
            metadata (dict[str, Any]): JSON-compatible description.
            references (dict[str, str]): Existing object references.
            files (dict[str, Path]): Artifact names and local sources.
            reservation_token (str | None): Capability for a previously reserved job draft.

        Returns:
            dict[str, Any]: Published manifest; failures retain the hidden draft.
        """
        path = f"/objects/{identifier(kind)}/{identifier(key)}"
        headers = {"X-Reservation-Token": reservation_token} if reservation_token else {}
        if reservation_token:
            self.json(
                "PATCH",
                f"/drafts/{kind}/{key}",
                json={"name": name, "metadata": metadata},
                headers=headers,
            )
        else:
            self.json(
                "POST",
                f"/objects/{kind}",
                json={"id": key, "name": name, "metadata": metadata, "references": references},
            )

        for filename, source in files.items():
            with source.open("rb") as stream:
                self.request(
                    "PUT", f"{path}/files/{identifier(filename)}", content=stream, headers=headers
                )

        return self.json("POST", f"{path}/publish", headers=headers)

    def download(self, kind: str, key: str, filename: str, target: Path) -> None:
        """Download and verify a file using its published checksum.

        Parameters:
            kind (str): Collection name.
            key (str): Published object ID.
            filename (str): Advertised artifact name.
            target (Path): Local destination.

        Returns:
            None: Writes verified bytes to the destination.
        """
        manifest = self.get(kind, key)
        content = self.request(
            "GET", f"/objects/{identifier(kind)}/{identifier(key)}/files/{identifier(filename)}"
        ).content

        expected = manifest["files"][filename]
        if (
            len(content) != expected["bytes"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Artifact checksum mismatch")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
