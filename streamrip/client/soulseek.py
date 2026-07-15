"""Soulseek client backed by the slskd daemon's REST API.

Unlike the streaming services, Soulseek is peer-to-peer: there are no stable
track/album IDs, no quality tiers, and downloads are indirect (slskd fetches
into its own download folder, from which we move the file into place).

Item ID format (opaque, round-trips through the CLI and database):

    track: "{username}::{urlsafe_b64(remote file path)}"
    album: "{username}::{urlsafe_b64(remote folder path)}"

Metadata is parsed from file/folder names (Soulseek carries no tags in its
search results), so it is best-effort; files keep their embedded tags.
"""

import asyncio
import base64
import logging
import os
import re
import shutil
import time
from pathlib import PurePosixPath, PureWindowsPath

from ..config import Config
from ..exceptions import NonStreamableError
from .client import Client
from .downloadable import Downloadable

logger = logging.getLogger("streamrip")

LOSSLESS_EXTS = {"flac", "alac", "ape", "wav", "aiff", "aif", "wv", "shn"}
LOSSY_EXTS = {"mp3", "m4a", "aac", "ogg", "opus", "wma"}
AUDIO_EXTS = LOSSLESS_EXTS | LOSSY_EXTS

# streamrip quality ladder for the soulseek source:
#   0: any audio (>=128kbps lossy ok)
#   1: lossy at >= min_bitrate (default 320)
#   2/3: FLAC (or other lossless)
#   4: any lossless (16 or 24 bit; convert with streamrip/beets if desired)
MAX_QUALITY = 4


def encode_item_id(username: str, remote_path: str) -> str:
    b64 = base64.urlsafe_b64encode(remote_path.encode()).decode()
    return f"{username}::{b64}"


def decode_item_id(item_id: str) -> tuple[str, str]:
    username, _, b64 = item_id.partition("::")
    if not b64:
        raise ValueError(f"Invalid soulseek item id: {item_id!r}")
    return username, base64.urlsafe_b64decode(b64.encode()).decode()


def _split_remote(path: str) -> list[str]:
    """Split a remote Soulseek path into components.

    Peers are usually Windows ('C:\\music\\Artist - Album\\01 song.flac') but
    unix-style paths appear too; handle both.
    """
    if "\\" in path:
        return list(PureWindowsPath(path).parts)
    return list(PurePosixPath(path).parts)


def remote_basename(path: str) -> str:
    parts = _split_remote(path)
    return parts[-1] if parts else path


def remote_dirname(path: str) -> str:
    """The remote parent folder path, preserving the peer's separator."""
    sep = "\\" if "\\" in path else "/"
    head, _, _ = path.rpartition(sep)
    return head


class SlskdClient:
    """Minimal async wrapper around the slskd REST API (aiohttp)."""

    def __init__(self, base_url: str, api_key: str, session):
        self.base = base_url.rstrip("/") + "/api/v0"
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.session = session

    async def _req(self, method: str, path: str, **kwargs):
        url = self.base + path
        async with self.session.request(
            method, url, headers=self.headers, **kwargs
        ) as resp:
            if resp.status == 401:
                raise NonStreamableError(
                    "slskd rejected the API key (401). Check soulseek.slskd_api_key."
                )
            resp.raise_for_status()
            if resp.status == 204 or resp.content_type != "application/json":
                return None
            return await resp.json()

    async def application_state(self) -> dict:
        return await self._req("GET", "/application")

    async def search(self, query: str, timeout_s: int) -> list[dict]:
        """Run a search to completion; return the list of peer responses."""
        req = await self._req(
            "POST",
            "/searches",
            json={"searchText": query, "searchTimeout": timeout_s * 1000},
        )
        search_id = req["id"]
        try:
            deadline = time.monotonic() + timeout_s + 15
            while time.monotonic() < deadline:
                state = await self._req("GET", f"/searches/{search_id}")
                if "Completed" in state.get("state", ""):
                    break
                await asyncio.sleep(1)
            return await self._req("GET", f"/searches/{search_id}/responses") or []
        finally:
            # keep slskd's search history tidy; ignore failures
            try:
                await self._req("DELETE", f"/searches/{search_id}")
            except Exception:
                pass

    async def directory(self, username: str, remote_dir: str) -> list[dict]:
        """List files in one folder of a peer's share. Returns file dicts."""
        resp = await self._req(
            "POST",
            f"/users/{username}/directory",
            json={"directory": remote_dir},
        )
        if resp is None:
            return []
        # returns a list of directories (usually one), each with "files"
        dirs = resp if isinstance(resp, list) else [resp]
        files = []
        for d in dirs:
            for f in d.get("files", []):
                # directory listings give bare filenames; qualify them
                name = f.get("filename", "")
                if "\\" not in name and "/" not in name:
                    sep = "\\" if "\\" in remote_dir else "/"
                    f = {**f, "filename": f"{remote_dir}{sep}{name}"}
                files.append(f)
        return files

    async def enqueue(self, username: str, files: list[dict]):
        """Queue downloads: files are [{filename, size}]."""
        await self._req(
            "POST",
            f"/transfers/downloads/{username}",
            json=[{"filename": f["filename"], "size": f["size"]} for f in files],
        )

    async def download_state(self, username: str, remote_path: str) -> dict | None:
        """Find the transfer record for a queued file, or None."""
        try:
            resp = await self._req("GET", f"/transfers/downloads/{username}")
        except Exception:
            return None
        for d in (resp or {}).get("directories", []):
            for f in d.get("files", []):
                if f.get("filename") == remote_path:
                    return f
        return None


def score_file(f: dict, prefs: list[str], min_bitrate: int) -> float:
    """Rank a search-result file: format preference, then quality signals."""
    ext = (f.get("extension") or "").lstrip(".").lower()
    if not ext:
        ext = remote_basename(f.get("filename", "")).rpartition(".")[2].lower()
    if ext not in AUDIO_EXTS or f.get("isLocked"):
        return -1
    try:
        fmt_rank = len(prefs) - prefs.index(ext)
    except ValueError:
        fmt_rank = 0.5 if ext in LOSSLESS_EXTS else 0.25
    score = fmt_rank * 1000
    if ext in LOSSLESS_EXTS:
        score += (f.get("bitDepth") or 16) + (f.get("sampleRate") or 44100) / 1e4
    else:
        br = f.get("bitRate") or 0
        if br and br < min_bitrate:
            return -1
        score += br / 10
    return score


def score_response(r: dict) -> float:
    """Rank a peer: free slot and speed matter, long queues hurt."""
    return (
        (1000 if r.get("hasFreeUploadSlot") else 0)
        + (r.get("uploadSpeed") or 0) / 1024
        - (r.get("queueLength") or 0) * 10
    )


# "01 - Artist - Title.flac" / "01 Title.flac" / "Artist - Title.flac" → title guess
TRACKNUM_RE = re.compile(r"^\s*\d{1,3}\s*[-._ ]+\s*")


def parse_track_name(filename: str) -> tuple[str | None, str]:
    """Best-effort (artist, title) from a remote filename."""
    stem = remote_basename(filename).rpartition(".")[0]
    stem = TRACKNUM_RE.sub("", stem)
    if " - " in stem:
        artist, _, title = stem.partition(" - ")
        return artist.strip() or None, title.strip()
    return None, stem.strip()


def parse_folder_name(folder: str) -> tuple[str | None, str, str | None]:
    """Best-effort (artist, album, year) from a remote folder path."""
    name = remote_basename(folder)
    year = None
    m = re.search(r"[([]?((?:19|20)\d{2})[)\]]?", name)
    if m:
        year = m.group(1)
        name = (name[: m.start()] + name[m.end() :]).strip(" -_([])")
    if " - " in name:
        artist, _, album = name.partition(" - ")
        return artist.strip() or None, album.strip(), year
    return None, name.strip(), year


class SoulseekClient(Client):
    source = "soulseek"
    max_quality = MAX_QUALITY
    logged_in = False

    def __init__(self, config: Config):
        self.global_config = config
        self.config = config.session.soulseek
        self.rate_limiter = self.get_rate_limiter(
            config.session.downloads.requests_per_minute,
        )

    async def login(self):
        self.session = await self.get_session(
            verify_ssl=self.global_config.session.downloads.verify_ssl
        )
        self.api = SlskdClient(
            self.config.slskd_url, self.config.slskd_api_key, self.session
        )
        state = await self.api.application_state()
        server = (state or {}).get("server", {})
        if not str(server.get("state", "")).count("Connected"):
            logger.warning(
                "slskd is reachable but not connected to the Soulseek network "
                "(server state: %s)",
                server.get("state"),
            )
        self.logged_in = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, media_type: str, query: str, limit: int = 50) -> list[dict]:
        assert media_type in ("track", "album"), f"Cannot search for {media_type}"
        responses = await self.api.search(query, self.config.search_timeout)
        prefs = [f.lstrip(".").lower() for f in self.config.preferred_formats]
        if media_type == "track":
            items = self._rank_tracks(responses, prefs)
        else:
            items = self._rank_albums(responses, prefs)
        return [{"collection": items[:limit]}] if items else []

    def _rank_tracks(self, responses: list[dict], prefs: list[str]) -> list[dict]:
        scored = []
        for r in responses:
            peer = score_response(r)
            for f in r.get("files", []):
                fs = score_file(f, prefs, self.config.min_bitrate)
                if fs < 0:
                    continue
                artist, title = parse_track_name(f["filename"])
                if artist is None:
                    # bare "03 - Title.flac": the parent folder usually
                    # carries the artist ("Artist - Album (year)")
                    artist, _, _ = parse_folder_name(
                        remote_dirname(f["filename"])
                    )
                scored.append(
                    (
                        fs * 1e6 + peer,
                        {
                            "id": encode_item_id(r["username"], f["filename"]),
                            "title": title,
                            "artist": artist or r["username"],
                            "username": r["username"],
                            "filename": f["filename"],
                            "size": f["size"],
                            "bitRate": f.get("bitRate"),
                            "bitDepth": f.get("bitDepth"),
                            "sampleRate": f.get("sampleRate"),
                            "length": f.get("length"),
                            "extension": remote_basename(f["filename"])
                            .rpartition(".")[2]
                            .lower(),
                        },
                    )
                )
        scored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in scored]

    def _rank_albums(self, responses: list[dict], prefs: list[str]) -> list[dict]:
        # Group each peer's files by remote folder; a folder of audio = an album.
        folders: dict[tuple[str, str], list[dict]] = {}
        peer_score: dict[str, float] = {}
        for r in responses:
            peer_score[r["username"]] = score_response(r)
            for f in r.get("files", []):
                folder = remote_dirname(f["filename"])
                if folder:
                    folders.setdefault((r["username"], folder), []).append(f)

        scored = []
        for (username, folder), files in folders.items():
            audio = [
                f for f in files if score_file(f, prefs, self.config.min_bitrate) >= 0
            ]
            if not audio:
                continue
            # folder quality = its typical file's score
            fscores = sorted(
                score_file(f, prefs, self.config.min_bitrate) for f in audio
            )
            median = fscores[len(fscores) // 2]
            artist, album, year = parse_folder_name(folder)
            scored.append(
                (
                    median * 1e6 + len(audio) * 1e3 + peer_score[username],
                    {
                        "id": encode_item_id(username, folder),
                        "title": album,
                        # dict form: AlbumSummary tries .get("name") on this
                        "artist": {"name": artist or username},
                        "year": year,
                        "username": username,
                        "folder": folder,
                        "tracks_count": len(audio),
                    },
                )
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in scored]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    async def get_metadata(self, item_id: str, media_type: str) -> dict:
        username, remote_path = decode_item_id(item_id)
        if media_type == "track":
            return await self._track_metadata(username, remote_path)
        if media_type == "album":
            return await self._album_metadata(username, remote_path)
        raise Exception(f"{media_type} not supported for soulseek")

    async def _track_metadata(self, username: str, remote_path: str) -> dict:
        artist, title = parse_track_name(remote_path)
        folder = remote_dirname(remote_path)
        folder_artist, album, year = parse_folder_name(folder) if folder else (
            None,
            "Unknown Album",
            None,
        )
        # size is needed to enqueue; fetch it from the folder listing
        size = None
        files = await self.api.directory(username, folder) if folder else []
        for f in files:
            if remote_basename(f["filename"]) == remote_basename(remote_path):
                size = f["size"]
                break
        if size is None:
            raise NonStreamableError(
                f"{remote_path} no longer listed by {username} on soulseek"
            )
        return {
            "id": encode_item_id(username, remote_path),
            "username": username,
            "filename": remote_path,
            "size": size,
            "title": title,
            "artist": artist or folder_artist or username,
            "album": album,
            "albumartist": folder_artist or artist or username,
            "year": year,
            "tracknumber": _tracknumber_of(remote_path),
        }

    async def _album_metadata(self, username: str, folder: str) -> dict:
        files = await self.api.directory(username, folder)
        prefs = [f.lstrip(".").lower() for f in self.config.preferred_formats]
        audio = sorted(
            (f for f in files if score_file(f, prefs, self.config.min_bitrate) >= 0),
            key=lambda f: remote_basename(f["filename"]),
        )
        if not audio:
            raise NonStreamableError(
                f"No downloadable audio in {folder} shared by {username}"
            )
        artist, album, year = parse_folder_name(folder)
        tracks = []
        for f in audio:
            t_artist, t_title = parse_track_name(f["filename"])
            tracks.append(
                {
                    "id": encode_item_id(username, f["filename"]),
                    "title": t_title,
                    "artist": t_artist or artist or username,
                    "filename": f["filename"],
                    "size": f["size"],
                }
            )
        return {
            "id": encode_item_id(username, folder),
            "username": username,
            "folder": folder,
            "title": album,
            "albumartist": artist or username,
            "year": year,
            "tracks": tracks,
            "tracktotal": len(tracks),
        }

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def get_downloadable(self, item_id: str, quality: int) -> Downloadable:
        username, remote_path = decode_item_id(item_id)
        folder = remote_dirname(remote_path)
        size = None
        for f in await self.api.directory(username, folder) if folder else []:
            if remote_basename(f["filename"]) == remote_basename(remote_path):
                size = f["size"]
                break
        if size is None:
            raise NonStreamableError(
                f"{remote_path} no longer listed by {username} on soulseek"
            )
        return SlskDownloadable(
            api=self.api,
            username=username,
            remote_path=remote_path,
            size=size,
            download_folder=self.config.download_folder,
            timeout_s=self.config.download_timeout,
        )


def _tracknumber_of(remote_path: str) -> int:
    m = re.match(r"^\s*(\d{1,3})", remote_basename(remote_path))
    return int(m.group(1)) if m else 1


class SlskDownloadable(Downloadable):
    """Queues a file on slskd, waits for it to finish, then moves it into place."""

    def __init__(
        self,
        api: SlskdClient,
        username: str,
        remote_path: str,
        size: int,
        download_folder: str,
        timeout_s: int,
    ):
        self.api = api
        self.username = username
        self.remote_path = remote_path
        self.download_folder = download_folder
        self.timeout_s = timeout_s
        self.session = api.session
        self.url = ""
        self.source = "soulseek"
        ext = remote_basename(remote_path).rpartition(".")[2].lower()
        self.extension = ext or "flac"
        self._size = size

    async def size(self) -> int:
        return self._size

    async def _download(self, path: str, callback):
        await self.api.enqueue(
            self.username, [{"filename": self.remote_path, "size": self._size}]
        )
        local = await self._wait_for_completion(callback)
        if not self.download_folder:
            raise NonStreamableError(
                "soulseek.download_folder is not set; cannot locate completed file"
            )
        if local is None:
            local = self._find_local_file()
        if local is None:
            raise NonStreamableError(
                f"Downloaded {self.remote_path} but could not find it under "
                f"{self.download_folder}"
            )
        shutil.move(local, path)

    async def _wait_for_completion(self, callback) -> str | None:
        deadline = time.monotonic() + self.timeout_s
        reported = 0
        while time.monotonic() < deadline:
            t = await self.api.download_state(self.username, self.remote_path)
            if t is not None:
                done = t.get("bytesTransferred", 0)
                if done > reported:
                    callback(done - reported)
                    reported = done
                state = t.get("state", "")
                if "Completed" in state:
                    if "Succeeded" in state:
                        return None  # find on disk
                    raise NonStreamableError(
                        f"soulseek transfer failed ({state}): "
                        f"{t.get('exception') or t.get('stateDescription') or ''}"
                    )
            await asyncio.sleep(2)
        raise NonStreamableError(
            f"soulseek transfer timed out after {self.timeout_s}s: {self.remote_path}"
        )

    def _find_local_file(self) -> str | None:
        """Locate the completed file under slskd's download folder.

        slskd saves to <downloads>/<derived subdirectory>/<basename>; the
        subdirectory template is user-configurable, so match on basename+size.
        """
        want_name = remote_basename(self.remote_path)
        for root, _, names in os.walk(self.download_folder):
            for name in names:
                if name == want_name:
                    p = os.path.join(root, name)
                    if os.path.getsize(p) == self._size:
                        return p
        return None
