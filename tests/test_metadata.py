from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def parse_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


class MetadataTests(unittest.TestCase):
    def test_metadata_required_fields(self) -> None:
        metadata = parse_simple_yaml(ROOT / "metadata.yaml")
        required = {
            "name",
            "display_name",
            "desc",
            "version",
            "astrbot_version",
            "author",
            "repo",
        }
        self.assertTrue(required.issubset(metadata))
        self.assertEqual(metadata["name"], "astrbot_plugin_llmpool")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertTrue(metadata["repo"].startswith("https://github.com/"))

    def test_plugin_schema_is_valid_json(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertIn("members", schema)
        self.assertIn("strategy", schema)
        self.assertNotIn("auto_create_pool_provider", schema)


if __name__ == "__main__":
    unittest.main()
