"""b2cviewer at /viewer/ beside the UI (`pipeline/viewer.py`).

The checkout is a stand-in: a `serve.py` with b2cviewer's two helpers
(`index`, `data_file`) and a page. What is tested is the wiring — that the
routes land before Gradio's catch-all mount, list this volume's runs by name
and serve only .glb files from under the output root.
"""
from __future__ import annotations

import os
import textwrap
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from fastapi.testclient import TestClient

    from pipeline import webui
except ImportError:  # the web UI's dependencies are not installed here
    webui = None

from pipeline.viewer import VIEWER_ENV, has_subject_file, viewer_link

SERVE_PY = textwrap.dedent('''
    from pathlib import Path

    def index(root, dirs=None, where=""):
        subjects = [{"name": name, "path": str(p.relative_to(root)), "rigged": False, "clips": []}
                    for name, d in dirs for p in sorted(d.glob("*.glb")) if not p.name.endswith(".clip.glb")]
        return {"root": str(root), "where": where, "subjects": subjects}

    def data_file(root, rel):
        p = (root / rel).resolve()
        return p if p.is_relative_to(root) and p.is_file() and p.suffix == ".glb" else None
''')


class TestViewerHelpers(unittest.TestCase):
    def test_link_is_relative_and_quoted(self):
        self.assertEqual(viewer_link("helical-1 a/b"), "viewer/?subject=helical-1%20a%2Fb")

    def test_a_clip_alone_is_not_a_subject_file(self):
        with TemporaryDirectory() as tmp:
            ply = Path(tmp) / "ply"
            ply.mkdir()
            (ply / "walk.clip.glb").write_bytes(b"x")
            self.assertFalse(has_subject_file(Path(tmp)))
            (ply / "scene.glb").write_bytes(b"x")
            self.assertTrue(has_subject_file(Path(tmp)))
            self.assertFalse(has_subject_file(None))


@unittest.skipIf(webui is None, "the web UI's dependencies are not installed here")
class TestViewerRoutes(unittest.TestCase):
    TOKEN = ""

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"
        self.checkout = Path(tmp.name) / "b2cviewer"
        (self.checkout / "web").mkdir(parents=True)
        (self.checkout / "serve.py").write_text(SERVE_PY)
        (self.checkout / "web" / "index.html").write_text("<!doctype html><title>b2cviewer</title>")
        run = self.data / "output" / "helical-20260929-000000-abc123" / "ply"
        run.mkdir(parents=True)
        (run / "scene.glb").write_bytes(b"glTF-subject")
        (run / "scene.ply").write_bytes(b"ply")
        patcher = unittest.mock.patch.dict(
            os.environ, {"B2C_DATA_DIR": str(self.data), VIEWER_ENV: str(self.checkout), "B2C_API_TOKEN": self.TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)
        for stale in ("B2C_OUTPUT_DIR", "B2C_UPLOAD_DIR", "B2C_RUN_JOBS_DIR", "B2C_LOG_DIR"):
            os.environ.pop(stale, None)
        self.client = TestClient(webui.build_server(gpu_count=1), raise_server_exceptions=False)

    def test_the_page_is_served_under_the_ui_mount(self):
        response = self.client.get("/viewer/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>b2cviewer</title>", response.text)
        self.assertEqual(self.client.get("/viewer", follow_redirects=False).headers["location"], "/viewer/")

    def test_the_index_lists_runs_by_name(self):
        index = self.client.get("/viewer/api/index").json()
        self.assertEqual(
            [(s["name"], s["path"]) for s in index["subjects"]],
            [("helical-20260929-000000-abc123", "helical-20260929-000000-abc123/ply/scene.glb")])

    def test_data_serves_glb_and_nothing_else(self):
        response = self.client.get("/viewer/data/helical-20260929-000000-abc123/ply/scene.glb")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"glTF-subject")
        self.assertEqual(response.headers["content-type"], "model/gltf-binary")
        for rel in ("helical-20260929-000000-abc123/ply/scene.ply", "../../b2cviewer/serve.py",
                    "helical-20260929-000000-abc123/ply/%2E%2E/%2E%2E/%2E%2E/b2cviewer/web/index.html"):
            self.assertEqual(self.client.get(f"/viewer/data/{rel}").status_code, 404, rel)

    def test_without_a_checkout_nothing_is_mounted(self):
        with unittest.mock.patch.dict(os.environ, {VIEWER_ENV: str(self.data / "missing")}):
            client = TestClient(webui.build_server(gpu_count=1), raise_server_exceptions=False)
        self.assertNotIn("/viewer/api/index", [getattr(r, "path", None) for r in client.app.routes])


@unittest.skipIf(webui is None, "the web UI's dependencies are not installed here")
class TestViewerBehindLogin(TestViewerRoutes):
    """With `B2C_API_TOKEN` set, `/viewer` takes the UI's Gradio session and nothing else.

    Inherits the routes' setUp with a token instead; the TestClient is plain
    http, so it carries Gradio's `access-token-unsecure-*` cookie — the one
    a LAN browser would."""

    TOKEN = "s3cret-token"

    def login(self):
        response = self.client.post("/login", data={"username": "b2c", "password": self.TOKEN})
        self.assertEqual(response.status_code, 200, response.text)

    def test_signed_out_the_page_goes_to_the_login_and_the_rest_is_refused(self):
        for page in ("/viewer", "/viewer/", "/viewer/index.html"):
            response = self.client.get(page, follow_redirects=False)
            self.assertEqual(response.status_code, 303, page)
            self.assertEqual(response.headers["location"], "/", page)
        for path in ("/viewer/api/index", "/viewer/data/helical-20260929-000000-abc123/ply/scene.glb",
                     "/viewer/main.js"):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_a_wrong_password_or_a_forged_cookie_opens_nothing(self):
        self.assertEqual(self.client.post("/login", data={"username": "b2c", "password": "nope"}).status_code, 400)
        self.client.cookies.set("access-token-unsecure-whatever", "forged")
        self.assertEqual(self.client.get("/viewer/api/index").status_code, 401)

    def test_logout_locks_the_viewer_again(self):
        self.login()
        self.assertEqual(self.client.get("/viewer/api/index").status_code, 200)
        self.client.get("/logout", follow_redirects=False)
        self.assertEqual(self.client.get("/viewer/api/index").status_code, 401)

    def test_the_ui_itself_is_not_behind_the_viewer_guard(self):
        self.assertNotEqual(self.client.get("/login_check").status_code, 303)

    # The inherited route tests, signed in.
    def test_the_page_is_served_under_the_ui_mount(self):
        self.login()
        super().test_the_page_is_served_under_the_ui_mount()

    def test_the_index_lists_runs_by_name(self):
        self.login()
        super().test_the_index_lists_runs_by_name()

    def test_data_serves_glb_and_nothing_else(self):
        self.login()
        super().test_data_serves_glb_and_nothing_else()


if __name__ == "__main__":
    unittest.main()
