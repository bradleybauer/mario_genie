import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.dataset_paths import find_chunk_files


class DatasetPathsTests(unittest.TestCase):
    def test_find_chunk_files_recurses_through_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            nested_dir = data_dir / "human_play"
            deeper_dir = data_dir / "other" / "session_a"
            nested_dir.mkdir(parents=True)
            deeper_dir.mkdir(parents=True)

            root_chunk = data_dir / "chunk_000000.npz"
            nested_chunk = nested_dir / "chunk_000001.npz"
            deeper_chunk = deeper_dir / "chunk_000002.npz"
            ignored_file = deeper_dir / "notes.txt"

            root_chunk.touch()
            nested_chunk.touch()
            deeper_chunk.touch()
            ignored_file.touch()

            self.assertEqual(
                find_chunk_files(data_dir),
                [str(root_chunk), str(nested_chunk), str(deeper_chunk)],
            )


if __name__ == "__main__":
    unittest.main()