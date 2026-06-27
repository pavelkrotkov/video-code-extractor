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
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def lesson_slug(url):
    """Last path segment of a lesson URL, sanitized for use in a filename."""
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")


def main():
    parser = argparse.ArgumentParser(description="Download lessons from a DeepLearning.AI course.")
    parser.add_argument(
        "course_url",
        help="The URL of the course page (e.g., https://learn.deeplearning.ai/courses/build-and-train-an-llm-with-jax)",
    )
    args = parser.parse_args()
    course_url = args.course_url

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
            full_url = requests.compat.urljoin(course_url, a["href"])
            lesson_links.append(full_url)

    # Deduplicate while preserving the order the lessons appear on the course page.
    # NOTE: do NOT sort() — lesson URLs end in /lesson/1, /lesson/2, ... /lesson/10,
    # and a lexicographic sort orders them 1, 10, 11, 2, 3, ... (and set() alone
    # discards page order entirely).
    lesson_links = list(dict.fromkeys(lesson_links))

    if not lesson_links:
        print(f"No lesson links containing '/lesson/' found at {course_url}")
        return

    # 2. For each lesson, fetch the page and extract the .m3u8 URL
    m3u8_pattern = re.compile(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)')  # adjust as needed

    outputs = []  # (index, m3u8_url)
    for i, lesson_url in enumerate(lesson_links):
        print(f"Processing lesson {i + 1}/{len(lesson_links)}: {lesson_url}")
        r = session.get(lesson_url)
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

    # 3. Stream-copy each HLS playlist into an .mp4 (no re-encode)
    # Reuse the same UA/cookies as the session so the CDN accepts the segment requests.
    ua = session.headers.get("User-Agent", "Mozilla/5.0")
    cookie = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    hdrs = f"User-Agent: {ua}\r\nReferer: {course_url}\r\n"
    if cookie:
        hdrs += f"Cookie: {cookie}\r\n"

    for idx, slug, m3u8_url in outputs:
        out = f"lesson_{idx:02d}_{slug}.mp4" if slug else f"lesson_{idx:02d}.mp4"
        print(f"Downloading {out} <- {m3u8_url}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-headers",
                hdrs,
                "-i",
                m3u8_url,
                "-c",
                "copy",
                "-bsf:a",
                "aac_adtstoasc",  # fix AAC ADTS -> ASC for MP4 container
                "-movflags",
                "+faststart",
                out,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
