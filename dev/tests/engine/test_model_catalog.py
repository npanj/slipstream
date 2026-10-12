import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from dev.tools import update_model_catalog
from install import catalog


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "cache/models.txt"
        self.bundled = self.root / "bundled.txt"
        self.bundled.write_text("company/Bundled\n")
        for name, value in (
            ("CACHE", self.output),
            ("BUNDLED", self.bundled),
            ("COLLECTION", "company/collection-0123"),
        ):
            patch = mock.patch.object(catalog, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(catalog.urllib.request, "urlopen")
        self.urlopen = patch.start()
        self.addCleanup(patch.stop)

    def response(self, payload):
        self.urlopen.return_value.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode()
        )

    def test_collection_produces_only_sorted_unique_model_ids(self):
        self.response(
            {
                "items": [
                    {"type": "model", "id": "community/Zeta"},
                    {"type": "dataset", "id": "org/examples"},
                    {"type": "model", "id": "company/Alpha"},
                    {"type": "model", "id": "community/Zeta"},
                    {"type": "model", "id": "../outside"},
                ]
            }
        )
        self.assertTrue(catalog.refresh())
        self.assertEqual(self.output.read_text(), "community/Zeta\ncompany/Alpha\n")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o644)
        request = self.urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            f"{catalog.HUB_ENDPOINT}/api/collections/{catalog.COLLECTION}",
        )
        self.assertEqual(
            catalog.official_ids(),
            ["community/Zeta", "company/Alpha", "company/Bundled"],
        )

    def test_repeated_refresh_succeeds_and_reports_destination_count(self):
        for _ in range(2):
            self.response({"items": [{"type": "model", "id": "company/Alpha"}]})
            with mock.patch("sys.stdout", io.StringIO()) as output:
                update_model_catalog.main(["--output", str(self.output)])
            self.assertIn("Wrote 1 official model IDs", output.getvalue())
            self.assertFalse(catalog.is_stale())
        os.utime(self.output, (0, 0))
        self.response({"items": [{"type": "model", "id": "company/Alpha"}]})
        self.assertEqual(catalog.main(["--output", str(self.output)]), 0)
        self.assertFalse(catalog.is_stale())

    def test_failed_or_invalid_collection_preserves_previous_snapshot(self):
        self.output.parent.mkdir()
        self.output.write_text("company/Existing\n")
        for payload in (
            {},
            [],
            {"items": None},
            {"items": []},
            {"items": [None, "invalid", {"type": "model", "id": "../outside"}]},
        ):
            with self.subTest(payload=payload):
                self.response(payload)
                self.assertFalse(catalog.refresh())
                self.assertEqual(self.output.read_text(), "company/Existing\n")
                self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        self.urlopen.side_effect = urllib.error.URLError("offline")
        with mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(catalog.main(["--output", str(self.output)]), 1)
        with self.assertRaises(SystemExit):
            update_model_catalog.main(["--output", str(self.output)])
        self.assertEqual(self.output.read_text(), "company/Existing\n")

    def test_corrupt_cache_can_be_replaced_without_losing_bundled_entries(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"\xff")
        self.assertEqual(catalog.official_ids(), ["company/Bundled"])
        self.response({"items": [{"type": "model", "id": "company/New"}]})
        self.assertTrue(catalog.refresh())
        self.assertEqual(catalog.official_ids(), ["company/Bundled", "company/New"])

    def test_without_a_collection_nothing_is_fetched_or_spawned(self):
        self.output.parent.mkdir()
        self.output.write_text("company/Existing\n")
        os.utime(self.output, (0, 0))
        with (
            mock.patch.object(catalog, "COLLECTION", None),
            mock.patch.object(catalog.subprocess, "Popen") as process,
        ):
            self.assertFalse(catalog.refresh())
            catalog.spawn_refresh()
            with mock.patch("sys.stderr", io.StringIO()):
                self.assertEqual(catalog.main(["--refresh"]), 0)
            with mock.patch("sys.stdout", io.StringIO()) as output:
                update_model_catalog.main(["--output", str(self.output)])
            self.assertIn("No model collection", output.getvalue())
        self.urlopen.assert_not_called()
        process.assert_not_called()
        self.assertEqual(self.output.read_text(), "company/Existing\n")
        self.assertEqual(
            catalog.official_ids(), ["company/Bundled", "company/Existing"]
        )

    def test_background_refresh_is_detached_and_fresh_cache_skips_spawn(self):
        with mock.patch.object(catalog.subprocess, "Popen") as process:
            catalog.spawn_refresh()
            self.assertIn("--refresh", process.call_args.args[0])
            self.assertTrue(process.call_args.kwargs["start_new_session"])
            self.assertTrue(process.call_args.kwargs["close_fds"])
            process.reset_mock()
            self.output.parent.mkdir()
            self.output.write_text("company/Existing\n")
            catalog.spawn_refresh()
            process.assert_not_called()
            self.output.unlink()
            process.side_effect = OSError("cannot spawn")
            catalog.spawn_refresh()


if __name__ == "__main__":
    unittest.main()
