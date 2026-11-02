"""WoGraph OSS remote bridge. No remote mutation by default."""
from __future__ import annotations

from urllib.parse import urlparse
import requests


class WoGraphOSS:
    def __init__(self, base_url: str, dataset_id: str, authorization: str = ""):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("A full, credential-free-in-URL HTTP(S) endpoint is required")
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Remote graph endpoints must use HTTPS")
        if not dataset_id or any(c not in "0123456789abcdef" for c in dataset_id.lower()) or len(dataset_id) > 48:
            raise ValueError("Use the hex dataset ID of an isolated workshop dataset")
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.authorization = authorization

    def _headers(self, content_type: str) -> dict:
        headers = {"Content-Type": content_type}
        if self.authorization:
            headers["Authorization"] = self.authorization
        return headers

    def inspect_graph(self) -> dict:
        """Read-only check that a configured workshop dataset is reachable."""
        response = requests.get(self.base_url + "/dataset/" + self.dataset_id,
                                headers=self._headers("application/json"), timeout=20)
        response.raise_for_status()
        dataset = response.json()
        return {"id": dataset.get("id"), "slug": dataset.get("slug"),
                "namespace": dataset.get("namespace")}

    def select(self, query: str) -> dict:
        response = requests.post(
            self.base_url + "/dataset/" + self.dataset_id + "/sparql",
            data=query.encode(), headers={**self._headers("application/sparql-query"),
                                           "Accept": "application/sparql-results+json"}, timeout=30)
        response.raise_for_status()
        return response.json()

    def import_synthetic_fixture(self, turtle: str, *, confirm_disposable_owned_dataset=False) -> dict:
        """Opt-in remote write. Always verify the target is a dedicated workshop dataset."""
        if not confirm_disposable_owned_dataset:
            raise PermissionError("Refusing write: confirm disposable owned workshop dataset explicitly")
        info = self.inspect_graph()
        if info["slug"] is None or not info["slug"].startswith("merlino-workshop-"):
            raise PermissionError("Refusing write: dataset slug must begin merlino-workshop-")
        response = requests.post(
            self.base_url + "/dataset/" + self.dataset_id + "/rdf", data=turtle.encode("utf-8"),
            headers=self._headers("text/turtle"), timeout=30)
        response.raise_for_status()
        return {"status": "imported", "dataset": self.dataset_id, "response_status": response.status_code}
