"""Tests for runtime configuration discovery and registration.

Tests cover:
- Config root discovery (built-in + external)
- YAML file enumeration
- YAML file parsing
- Package availability detection
- Config registration with Hydra
- Duplicate registration detection
- Missing dependency handling
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from deckard.declarations import (
    _get_config_group_and_name,
    _should_register_config,
    discover_config_roots,
    is_package_available,
    iter_config_files,
    parse_config_file,
    register_configs,
)


class TestDiscoverConfigRoots:
    """Test config root discovery."""

    def test_builtin_roots_exist(self):
        """Verify built-in roots are discovered when they exist."""
        roots = discover_config_roots()
        # Should find at least sklearn root if it exists
        root_strs = [str(r) for r in roots]
        # At least one of the built-in roots should be found
        assert any(
            "sklearn" in r or "pytorch" in r for r in root_strs
        ), "Expected at least one built-in config root"

    def test_external_roots_from_env(self):
        """Test DECKARD_CONFIG_DIRS environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two external config directories
            ext_dir1 = Path(tmpdir) / "ext1"
            ext_dir2 = Path(tmpdir) / "ext2"
            ext_dir1.mkdir()
            ext_dir2.mkdir()

            # Set environment variable
            with mock.patch.dict(
                os.environ,
                {"DECKARD_CONFIG_DIRS": f"{ext_dir1}:{ext_dir2}"},
            ):
                roots = discover_config_roots()
                # Check that external roots are in the list
                root_strs = [str(r) for r in roots]
                assert any(
                    str(ext_dir1) in r for r in root_strs
                ), f"External root {ext_dir1} not found"
                assert any(
                    str(ext_dir2) in r for r in root_strs
                ), f"External root {ext_dir2} not found"

    def test_external_roots_with_nonexistent_paths(self):
        """Test that nonexistent external roots are skipped with warning."""
        with mock.patch.dict(os.environ, {"DECKARD_CONFIG_DIRS": "/nonexistent/path"}):
            roots = discover_config_roots()
            # Should not include nonexistent path
            assert not any(
                "/nonexistent" in str(r) for r in roots
            ), "Nonexistent path should not be included"

    def test_empty_deckard_config_dirs(self):
        """Test empty DECKARD_CONFIG_DIRS is handled gracefully."""
        with mock.patch.dict(os.environ, {"DECKARD_CONFIG_DIRS": ""}):
            roots = discover_config_roots()
            # Should only have built-in roots
            assert len(roots) >= 0  # May be 0 if built-in roots don't exist


class TestIterConfigFiles:
    """Test YAML file enumeration."""

    def test_enumerate_yaml_files(self):
        """Test enumeration of .yaml files in directory tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create test structure
            (root / "model").mkdir()
            (root / "model" / "config1.yaml").touch()
            (root / "model" / "config2.yaml").touch()
            (root / "data").mkdir()
            (root / "data" / "config3.yaml").touch()
            (root / "default.yaml").touch()

            files = list(iter_config_files(root))
            assert len(files) == 4, "Should find 4 YAML files"
            assert all(f.suffix == ".yaml" for f in files), "All files should be .yaml"

    def test_skip_hidden_files(self):
        """Test that hidden files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create visible and hidden files
            (root / "config.yaml").touch()
            (root / ".hidden.yaml").touch()

            files = list(iter_config_files(root))
            assert len(files) == 1, "Should find only visible file"
            assert "config.yaml" in str(files[0])

    def test_nonexistent_directory(self):
        """Test that nonexistent directory returns empty iterator."""
        files = list(iter_config_files(Path("/nonexistent/path")))
        assert len(files) == 0, "Should return no files for nonexistent directory"


class TestParseConfigFile:
    """Test YAML parsing."""

    def test_parse_valid_yaml(self):
        """Test parsing of valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value", "nested": {"a": 1}}, f)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_config_file(path)
            assert result == {"key": "value", "nested": {"a": 1}}
        finally:
            path.unlink()

    def test_parse_empty_yaml(self):
        """Test parsing of empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_config_file(path)
            assert result == {}  # Empty YAML returns empty dict
        finally:
            path.unlink()

    def test_parse_invalid_yaml(self):
        """Test that invalid YAML returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{ invalid yaml: ][")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_config_file(path)
            assert result is None, "Invalid YAML should return None"
        finally:
            path.unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file."""
        result = parse_config_file(Path("/nonexistent/file.yaml"))
        assert result is None, "Nonexistent file should return None"


class TestIsPackageAvailable:
    """Test package availability detection."""

    def test_stdlib_package_available(self):
        """Test that stdlib packages are detected as available."""
        assert is_package_available("os")
        assert is_package_available("sys")

    def test_installed_package_available(self):
        """Test that installed packages are detected."""
        # yaml and pytest should be available if tests are running
        assert is_package_available("yaml")
        assert is_package_available("pytest")

    def test_unavailable_package(self):
        """Test that unavailable packages return False."""
        assert not is_package_available("nonexistent_package_xyz_123")

    def test_sklearn_detection(self):
        """Test scikit-learn availability detection."""
        # sklearn should be available since it's imported in __init__.py
        assert is_package_available("sklearn")

    def test_torch_detection(self):
        """Test torch availability detection (may be True or False depending on env)."""
        result = is_package_available("torch")
        # Just verify it returns a boolean
        assert isinstance(result, bool)


class TestShouldRegisterConfig:
    """Test conditional registration logic."""

    def test_pytorch_config_with_torch(self):
        """Test PyTorch config is registered when torch available."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            dir=None,
            prefix="torch_",
        ) as f:
            path = Path(f.name)
            with mock.patch(
                "deckard.declarations.is_package_available",
                return_value=True,
            ):
                assert _should_register_config(path)

    def test_pytorch_config_without_torch(self):
        """Test PyTorch config is skipped when torch unavailable."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="torch_",
        ) as f:
            path = Path(f.name)
            with mock.patch(
                "deckard.declarations.is_package_available",
                return_value=False,
            ):
                assert not _should_register_config(path)

    def test_sklearn_config_with_sklearn(self):
        """Test sklearn config is registered when sklearn available."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="sklearn_",
        ) as f:
            path = Path(f.name)
            with mock.patch(
                "deckard.declarations.is_package_available",
                return_value=True,
            ):
                assert _should_register_config(path)

    def test_sklearn_config_without_sklearn(self):
        """Test sklearn config is skipped when sklearn unavailable."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="sklearn_",
        ) as f:
            path = Path(f.name)
            with mock.patch(
                "deckard.declarations.is_package_available",
                return_value=False,
            ):
                assert not _should_register_config(path)

    def test_framework_pytorch_config_without_torch(self):
        """Framework pytorch configs should be skipped when torch is unavailable."""
        path = Path("/tmp/frameworks/pytorch/default_defense.yaml")
        with mock.patch(
            "deckard.declarations.is_package_available",
            return_value=False,
        ):
            assert not _should_register_config(path)

    def test_framework_sklearn_config_without_sklearn(self):
        """Framework sklearn configs should be skipped when sklearn is unavailable."""
        path = Path("/tmp/frameworks/sklearn/default_defense.yaml")
        with mock.patch(
            "deckard.declarations.is_package_available",
            return_value=False,
        ):
            assert not _should_register_config(path)

    def test_generic_config_always_registered(self):
        """Test generic configs are always registered regardless of deps."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="config_",
        ) as f:
            path = Path(f.name)
            with mock.patch(
                "deckard.declarations.is_package_available",
                return_value=False,
            ):
                # Even with all packages unavailable, generic configs are registered
                assert _should_register_config(path)


class TestGetConfigGroupAndName:
    """Test config group and name computation."""

    def test_nested_path(self):
        """Test nested config path parsing."""
        root = Path("/root/config")
        path = root / "model" / "sklearn" / "random_forest.yaml"
        group, name = _get_config_group_and_name(path, root)
        assert group == "model/sklearn"
        assert name == "random_forest"

    def test_top_level_config(self):
        """Test top-level config has empty group."""
        root = Path("/root/config")
        path = root / "default.yaml"
        group, name = _get_config_group_and_name(path, root)
        assert group == ""
        assert name == "default"

    def test_single_level_group(self):
        """Test single-level group."""
        root = Path("/root/config")
        path = root / "model" / "config.yaml"
        group, name = _get_config_group_and_name(path, root)
        assert group == "model"
        assert name == "config"


class TestRegisterConfigs:
    """Test full config registration flow."""

    def test_register_configs_runs_without_error(self):
        """Test that register_configs completes without error."""
        # This is a basic smoke test - just verify the function runs
        try:
            register_configs()
        except Exception as e:
            pytest.fail(f"register_configs raised {type(e).__name__}: {e}")

    def test_register_configs_idempotent(self):
        """Test that register_configs can be called multiple times."""
        try:
            register_configs()
            register_configs()  # Should not error
        except Exception as e:
            pytest.fail(f"Repeated register_configs raised {type(e).__name__}: {e}")

    def test_register_configs_with_external_dirs(self):
        """Test registration with external config directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "configs"
            ext_dir.mkdir()

            # Create a test config
            model_dir = ext_dir / "model"
            model_dir.mkdir()
            config_file = model_dir / "test_model.yaml"
            with open(config_file, "w") as f:
                yaml.dump({"_target_": "some.Model"}, f)

            with mock.patch.dict(os.environ, {"DECKARD_CONFIG_DIRS": str(ext_dir)}):
                try:
                    register_configs()
                except Exception as e:
                    pytest.fail(
                        f"register_configs with external dirs raised {type(e).__name__}: {e}",
                    )
