# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests",
#     "beautifulsoup4",
# ]
# ///
import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def lesson_slug(url):
    """Last path segment of a lesson URL, sanitized for use in a filename."""
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")


def course_slug(url):
    """Short identifier derived from the last path segment of the course URL."""
    path = urlparse(url).path.rstrip("/")
    seg = path.rsplit("/", 1)[-1] if path else ""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", seg).strip("-") or "course"


def get_duration(url_or_path, hdrs_str=""):
    """Return duration in seconds via ffprobe, or None on any error."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
    ]
    if hdrs_str:
        cmd += ["-headers", hdrs_str]
    cmd.append(str(url_or_path))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def _hms_to_secs(hms):
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"unexpected time format: {hms!r}")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def _fmt_time(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_size(nbytes):
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    if nbytes >= 1e3:
        return f"{nbytes / 1e3:.1f} KB"
    return f"{nbytes} B"


def _fmt_dur_hm(secs):
    h, rem = divmod(int(secs), 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


_TIME_RE = re.compile(r"time=(\d+:\d+:\d+(?:\.\d+)?)")


def download_with_progress(m3u8_url, out_path, idx, total, slug, hdrs):
    """Stream-copy an HLS playlist to out_path, printing a live progress line."""
    print(f"[{idx}/{total}] {slug}")
    total_secs = get_duration(m3u8_url, hdrs)

    cmd = [
        "ffmpeg",
        "-y",
        "-headers",
        hdrs,
        "-i",
        m3u8_url,
        "-c",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stderr or []:
        m = _TIME_RE.search(line)
        if m:
            try:
                elapsed = _hms_to_secs(m.group(1))
            except (ValueError, TypeError):
                continue
            if sys.stdout.isatty():
                if total_secs:
                    pct = min(100, int(elapsed / total_secs * 100))
                    print(
                        f"\r  {pct:3d}%  {_fmt_time(elapsed)} / {_fmt_time(total_secs)}",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {_fmt_time(elapsed)} elapsed", end="", flush=True)
    proc.wait()
    print()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ffmpeg")


def merge_lessons(raw_dir, lesson_paths, merged_path):
    """Concatenate lesson MP4s into a single file, then remove the individual files."""
    concat_file = raw_dir / f"concat_{merged_path.stem}.txt"
    lines = []
    for p in lesson_paths:
        # as_posix() avoids Windows backslash issues; escape single quotes for ffmpeg's parser
        safe = p.resolve().as_posix().replace("'", "\\'")
        lines.append(f"file '{safe}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(merged_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if not merged_path.exists() or merged_path.stat().st_size == 0:
            raise OSError("merged file is empty or missing")
    finally:
        concat_file.unlink(missing_ok=True)

    # Delete sources separately: a failure here doesn't invalidate the merged file,
    # so warn per file rather than treating it as a merge failure.
    for p in lesson_paths:
        try:
            p.unlink()
        except OSError as exc:
            print(f"Warning: could not remove {p}: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Download lessons from a DeepLearning.AI course.")
    parser.add_argument(
        "course_url",
        help="The URL of the course page (e.g., https://learn.deeplearning.ai/courses/build-and-train-an-llm-with-jax)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="keep individual lesson files instead of merging into one",
    )
    args = parser.parse_args()
    course_url = args.course_url
    slug = course_slug(course_url)

    raw_dir = Path("raw")
    raw_dir.mkdir(exist_ok=True)

    # Load session cookies from your browser (export them with an extension or manually)
    session = requests.Session()
    # Optionally set cookie from browser's "document.cookie" after logging in
    # session.cookies.set(...)

    # 1. Get list of lesson URLs
    resp = session.get(course_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Find all links to lessons – look for anchor tags with href containing '/lesson/'
    lesson_links = []
    for a in soup.find_all("a", href=True):
        if "/lesson/" in a["href"]:
            full_url = urljoin(course_url, a["href"])
            lesson_links.append(full_url)

    # Deduplicate while preserving the order the lessons appear on the course page.
    # NOTE: do NOT sort() — lesson URLs end in /lesson/1, /lesson/2, ... /lesson/10,
    # and a lexicographic sort orders them 1, 10, 11, 2, 3, ... (and set() alone
    # discards page order entirely).
    lesson_links = list(dict.fromkeys(lesson_links))

    if not lesson_links:
        sys.exit(
            f"Error: No lesson links containing '/lesson/' found at {course_url}.\n"
            "Please ensure you have configured your session cookies in the script if the course requires authentication."
        )

    # 2. For each lesson, fetch the page and extract the .m3u8 URL
    m3u8_pattern = re.compile(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)')  # adjust as needed

    outputs = []  # (index, slug, m3u8_url)
    for i, lesson_url in enumerate(lesson_links):
        print(f"Processing lesson {i + 1}/{len(lesson_links)}: {lesson_url}")
        r = session.get(lesson_url)
        r.raise_for_status()
        # Search for m3u8 in the HTML (sometimes it's in a <script>)
        match = m3u8_pattern.search(r.text)
        if not match:
            # Fallback: maybe it's inside a data attribute or JSON
            # Try to find a known pattern like "videoUrl":"..."
            match = re.search(r'["\']videoUrl["\']\s*:\s*["\']([^"\']+m3u8)', r.text)
        if match:
            m3u8_url = match.group(1)
            print(f"  Found m3u8: {m3u8_url}")
            outputs.append((i + 1, lesson_slug(lesson_url), m3u8_url))
        else:
            print("  WARNING: Could not find m3u8 URL")

    if not outputs:
        sys.exit(
            "Error: No m3u8 video URLs could be found in any of the lessons.\n"
            "Your session cookies may have expired or are invalid for the lesson pages."
        )

    # 3. Stream-copy each HLS playlist into an .mp4 (no re-encode)
    # Reuse the same UA/cookies as the session so the CDN accepts the segment requests.
    ua = session.headers.get("User-Agent", "Mozilla/5.0")
    cookie = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    hdrs = f"User-Agent: {ua}\r\nReferer: {course_url}\r\n"
    if cookie:
        hdrs += f"Cookie: {cookie}\r\n"

    downloaded: list[Path] = []
    for idx, lslug, m3u8_url in outputs:
        out_path = raw_dir / (f"lesson_{idx:02d}_{lslug}.mp4" if lslug else f"lesson_{idx:02d}.mp4")
        try:
            download_with_progress(
                m3u8_url, out_path, idx, len(outputs), lslug or f"lesson_{idx:02d}", hdrs
            )
            downloaded.append(out_path)
        except FileNotFoundError:
            sys.exit(
                "Error: 'ffmpeg' is not installed or not found in your PATH. "
                "Please install ffmpeg to download lessons."
            )
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: ffmpeg failed for lesson {idx} (exit code {e.returncode}). Skipping...",
                file=sys.stderr,
            )

    if not downloaded:
        sys.exit("Error: No lessons were downloaded successfully.")

    # 4. Optionally merge all lessons into one file
    merged_path = None
    # Compare against lesson_links (not outputs) so lessons that had no m3u8 also count as missing
    partial = len(downloaded) < len(lesson_links)
    if not args.no_merge and partial:
        print(
            f"Warning: only {len(downloaded)}/{len(outputs)} lessons downloaded; "
            "skipping merge to avoid an incomplete combined file. "
            "Re-run with --no-merge to suppress this check.",
            file=sys.stderr,
        )
    if not args.no_merge and not partial:
        merged_path = raw_dir / f"{slug}.mp4"
        print(f"Merging {len(downloaded)} lesson(s) -> {merged_path}")
        try:
            merge_lessons(raw_dir, downloaded, merged_path)
        except (subprocess.CalledProcessError, OSError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                err_msg = f"exit code {e.returncode}"
            else:
                err_msg = str(e)
            print(
                f"Warning: merge failed ({err_msg}). Individual files kept.",
                file=sys.stderr,
            )
            # Remove any partial output so a corrupt file isn't mistaken for a successful merge
            if merged_path.exists():
                merged_path.unlink()
            merged_path = None

    # 5. Print final stats
    print()
    if merged_path and merged_path.exists():
        total_bytes = merged_path.stat().st_size
        duration = get_duration(merged_path)
        dur_str = f", {_fmt_dur_hm(duration)}" if duration else ""
        print(f"Downloaded {len(downloaded)} lesson(s) ({_fmt_size(total_bytes)} total)")
        print(f"Merged: {merged_path} ({_fmt_size(total_bytes)}{dur_str})")
    else:
        total_bytes = sum(p.stat().st_size for p in downloaded if p.exists())
        print(f"Downloaded {len(downloaded)} lesson(s) ({_fmt_size(total_bytes)} total)")
        for p in downloaded:
            if p.exists():
                print(f"  {p} ({_fmt_size(p.stat().st_size)})")


if __name__ == "__main__":
    main()
