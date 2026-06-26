"""Stage 2 — drop near-duplicate frames (perceptual hash).

Adjacent sampled frames are usually identical (the screen hasn't changed between two 1-fps
samples), so OCR-ing all of them wastes time and produces redundant extractions. This stage
collapses each *run* of perceptually near-identical neighbours down to its first frame, keeping
that frame's original timestamp as the provenance for the whole run.

"Near-identical" is measured by the Hamming distance between perceptual hashes (``imagehash``):
the number of differing bits between two image fingerprints. ``0`` means the hashes are equal;
larger values mean more visual change. Two frames are treated as duplicates when their distance
is ``<= max_distance``. The default of ``4`` tolerates compression noise and a blinking cursor
while still splitting on a real content change.

``imagehash`` is imported lazily (it pulls in numpy/scipy) and the hash function is injectable so
the run-collapsing logic can be unit-tested without touching the disk or the hashing library.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vce.types import Frame

if TYPE_CHECKING:
    from imagehash import ImageHash

# A computed perceptual hash. ``imagehash.ImageHash`` supports ``a - b`` for Hamming distance.
Hash = Any


def _phash(frame: Frame) -> ImageHash:
    """Default hash function: the perceptual hash of the image at ``frame.path``.

    Opens the frame with Pillow and returns an ``imagehash.ImageHash``; two such hashes can be
    subtracted (``a - b``) to get the Hamming distance between them.

    Raises a clear :class:`ValueError` if the image cannot be read (missing path, permission
    error, or unrecognized format) so a bad frame surfaces a useful message instead of a raw
    ``OSError`` deep inside the dedup loop.
    """
    import imagehash
    from PIL import Image

    try:
        with Image.open(frame.path) as img:
            return imagehash.phash(img)
    except Exception as exc:
        # This helper does exactly one risky thing — open and hash an image — so wrap *any*
        # failure (missing path, permission, unrecognized format, or a decoder/imagehash error
        # on corrupt pixel data) into a clear ValueError instead of leaking it into the loop.
        raise ValueError(f"cannot open frame image at {frame.path}: {exc}") from exc


def dedup_frames(
    frames: Iterable[Frame],
    *,
    max_distance: int = 4,
    hash_func: Callable[[Frame], Hash] = _phash,
) -> list[Frame]:
    """Return ``frames`` with perceptually near-identical neighbours removed.

    Keeps the first frame of each run of near-duplicates, comparing every frame against the
    *last kept* frame: a frame is dropped when ``hash(frame) - hash(last_kept) <= max_distance``
    (Hamming distance over perceptual hashes). Input order and the kept frames' original
    timestamps are preserved.

    Args:
        frames: Sampled frames in timeline order.
        max_distance: Maximum perceptual-hash Hamming distance for two frames to count as
            duplicates. ``0`` keeps only exact-hash matches together; larger values are more
            tolerant of compression noise and minor motion.
        hash_func: Computes the perceptual hash of a frame. Injectable for testing; defaults to
            :func:`_phash`, which reads the image from ``frame.path`` and returns an
            ``imagehash.ImageHash``. The returned hashes must support ``a - b`` (Hamming distance).
    """
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    kept: list[Frame] = []
    last_hash: Hash | None = None
    for frame in frames:
        frame_hash = hash_func(frame)
        if last_hash is not None and (frame_hash - last_hash) <= max_distance:
            continue  # near-duplicate of the last kept frame; drop it, keep the run's first
        kept.append(frame)
        last_hash = frame_hash
    return kept
