"""Read aligned source and target text files."""

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


@dataclass(frozen=True)
class ParallelTextStats:
    """Counts from a successfully read parallel corpus."""

    total_pairs: int
    skipped_empty_pairs: int

    @property
    def kept_pairs(self) -> int:
        return self.total_pairs - self.skipped_empty_pairs


class ParallelTextReader:
    """Read one aligned sentence pair per line without changing the raw files."""

    def __init__(self, source_path: str | Path, target_path: str | Path):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.stats: ParallelTextStats | None = None

    def read(self) -> list[tuple[str, str]]:
        """Return nonempty pairs; fail if either file has unmatched lines.

        Empty pairs are removed together, preserving alignment. Statistics
        become available only after the complete corpus passes validation.
        """
        self.stats = None
        pairs: list[tuple[str, str]] = []
        total = skipped = 0
        with (
            self.source_path.open(encoding="utf-8-sig") as source_file,
            self.target_path.open(encoding="utf-8-sig") as target_file,
        ):
            for line_number, (source, target) in enumerate(
                zip_longest(source_file, target_file), start=1
            ):
                if source is None or target is None:
                    raise ValueError(
                        f"Parallel files have different line counts at line {line_number}: "
                        f"{self.source_path} and {self.target_path}"
                    )
                total += 1
                source, target = source.strip(), target.strip()
                if not source or not target:
                    skipped += 1
                    continue
                pairs.append((source, target))

        self.stats = ParallelTextStats(total, skipped)
        return pairs
