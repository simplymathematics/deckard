import collections
import unittest
import warnings
from pathlib import Path

import yaml

from deckard.base.hashable import (
    BaseHashable,
    from_dict,
    from_yaml,
    generate_grid_search,
    generate_line_search,
    generate_queue,
    my_hash,
    sort_queue,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestHashable(
    collections.namedtuple(
        typename="TestHashable",
        field_names="files ",
        defaults=({}),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))


class testBaseHashable(unittest.TestCase):
    def setUp(self):
        self.path = "tmp"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.filetype = "yaml"
        self.config = """
        !TestHashable
            files:
                path : tmp
                filetype : yaml
        """
        self.config2 = """
        !TestHashable
            files:
                path : tmp
                filetype : yaml
        """
        self.config3 = """
        !TestHashable
            files:
                path : tmp
                filetype : yml
        """
        self.config4 = """
            files:
                path : tmp
                filetype : yaml
        """
        yaml.add_constructor("!TestHashable", TestHashable)
        self.hashable = yaml.load(self.config, Loader=yaml.FullLoader)
        self.hashable2 = yaml.load(self.config2, Loader=yaml.FullLoader)
        self.hashable3 = yaml.load(self.config3, Loader=yaml.FullLoader)
        document = "!TestHashable\n" + self.config4
        self.hashable4 = yaml.load(document, Loader=yaml.FullLoader)
        self.list = self.path + "/list_test.yaml"
        with open(self.list, "w") as f:
            yaml.dump(
                [
                    {
                        "files": {
                            "filetype": "json",
                            "path": "foo",
                        },
                    },
                    {
                        "files": {
                            "filetype": "csv",
                            "path": "foo",
                        },
                    },
                    {
                        "files": {
                            "filetype": "xml",
                            "path": "foo",
                        },
                    },
                ],
                f,
            )

    def test_new(self):
        self.assertIsInstance(self.hashable, TestHashable)

    def test_hash(self):
        self.assertEqual(my_hash(self.hashable), my_hash(self.hashable2))
        self.assertNotEqual(my_hash(self.hashable), my_hash(self.hashable3))

    def test_repr(self):
        self.assertIsInstance(str(self.hashable), str)

    def test_to_dict(self):
        self.assertIsInstance(self.hashable.to_dict(), dict)

    def test_to_yaml(self):
        string_ = self.hashable.to_yaml()
        with open(self.path + "/test.yaml", "w") as f:
            yaml.dump(string_, f)
        with open(self.path + "/test.yaml", "r") as f:
            self.assertEqual(yaml.load(f, Loader=yaml.FullLoader), string_)

    def test_from_yaml(self):
        string_ = self.hashable.to_yaml()
        with open(self.path + "/test.yaml", "w") as f:
            yaml.dump(string_, f)
        hashable = from_yaml(hashable=self.hashable, filename=self.path + "/test.yaml")
        self.assertEqual(hashable, self.hashable)

    def test_save_yaml_load(self):
        filename = self.hashable.save_yaml(self.path)
        with open(filename, "r") as f:
            self.assertEqual(
                yaml.load(f, Loader=yaml.FullLoader),
                self.hashable.to_dict(),
            )

    def test_save_yaml(self):
        filename = self.hashable.save_yaml(path=self.path)
        test_filename = Path(self.path) / Path(
            my_hash(self.hashable) + "." + self.filetype,
        )
        self.assertEqual(filename, test_filename)

    def test_set_param(self):
        new = self.hashable.set_param("files.path", "foo")
        self.assertEqual(new.files["path"], "foo")

    def test_generate_line_search(self):
        exp_list = generate_line_search(
            self.hashable,
            "files.filetype",
            ["json", "csv", "xml"],
        )
        self.assertEqual(
            exp_list[0].to_dict(),
            {
                "files": {
                    "filetype": "json",
                    "path": "tmp",
                },
            },
        )
        self.assertEqual(
            exp_list[1].to_dict(),
            {
                "files": {
                    "filetype": "csv",
                    "path": "tmp",
                },
            },
        )
        self.assertEqual(
            exp_list[2].to_dict(),
            {
                "files": {
                    "filetype": "xml",
                    "path": "tmp",
                },
            },
        )

    def test_generate_grid_search(self):
        grid = {
            "files.filetype": ["json", "csv", "xml"],
            "files.path": ["foo", "bar", "baz"],
        }
        exp_list = generate_grid_search(self.hashable, grid)
        self.assertEqual(
            exp_list[0].to_dict(),
            {
                "files": {
                    "filetype": "json",
                    "path": "foo",
                },
            },
        )
        self.assertEqual(
            exp_list[1].to_dict(),
            {
                "files": {
                    "filetype": "json",
                    "path": "bar",
                },
            },
        )
        self.assertEqual(
            exp_list[2].to_dict(),
            {
                "files": {
                    "filetype": "json",
                    "path": "baz",
                },
            },
        )
        self.assertEqual(
            exp_list[3].to_dict(),
            {
                "files": {
                    "filetype": "csv",
                    "path": "foo",
                },
            },
        )
        self.assertEqual(
            exp_list[4].to_dict(),
            {
                "files": {
                    "filetype": "csv",
                    "path": "bar",
                },
            },
        )
        self.assertEqual(
            exp_list[5].to_dict(),
            {
                "files": {
                    "filetype": "csv",
                    "path": "baz",
                },
            },
        )
        self.assertEqual(
            exp_list[6].to_dict(),
            {
                "files": {
                    "filetype": "xml",
                    "path": "foo",
                },
            },
        )
        self.assertEqual(
            exp_list[7].to_dict(),
            {
                "files": {
                    "filetype": "xml",
                    "path": "bar",
                },
            },
        )
        self.assertEqual(
            exp_list[8].to_dict(),
            {
                "files": {
                    "filetype": "xml",
                    "path": "baz",
                },
            },
        )

    def test_generate_queue(self):
        grid = {"files.filetype": ["json", "csv", "xml"]}
        queue_file = generate_queue(self.hashable, grid, path=Path(self.path) / "queue")
        self.assertTrue(queue_file.exists())
        grid = {"files.filetype": ["json", "csv", "xml"]}
        queue_file = generate_queue(self.hashable, grid, path=Path(self.path) / "queue")
        with open(queue_file, "r") as f:
            lines = f.readlines()
            count = len(lines)
        self.assertEqual(count, 4)  # 3 + 1 header line

    def test_sort_queue(self):
        grid = {"files.filetype": ["json", "csv", "xml"]}
        _ = generate_queue(self.hashable, grid, path=Path(self.path) / "queue")
        new_queue_file = sort_queue(path=Path(self.path) / "queue")
        self.assertTrue(new_queue_file.exists())

    def test_from_dict(self):
        config = {
            "files": {
                "filetype": "json",
                "path": "tmp",
            },
        }
        hashable = from_dict(hashable=self.hashable, config=config)
        self.assertEqual(hashable.to_dict(), config)

    def tearDown(self):
        import shutil

        if Path(self.path).exists():
            shutil.rmtree(self.path)
