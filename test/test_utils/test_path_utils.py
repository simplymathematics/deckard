from deckard.path_utils import ensure_parent_dir, safe_unlink, to_posix_path


def test_to_posix_path_normalizes_windows_separators():
    assert to_posix_path(r"nested\dir\file.txt") == "nested/dir/file.txt"


def test_ensure_parent_dir_creates_parent_directory(tmp_path):
    path = ensure_parent_dir(tmp_path / "nested" / "artifact.json")

    assert path == tmp_path / "nested" / "artifact.json"
    assert path.parent.exists()


def test_safe_unlink_deletes_existing_file_and_ignores_missing(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")

    safe_unlink(path)
    safe_unlink(path)

    assert not path.exists()
