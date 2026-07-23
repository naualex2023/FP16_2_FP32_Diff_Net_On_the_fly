#!/usr/bin/env python3
"""
test_model_resolver.py — Tests for the generic HF / local model resolver.

These tests do NOT require network access or CUDA; they cover the pure-Python
detection / path-resolution logic and architecture inference.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure the repo root is importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from model_resolver import (
    is_hf_repo_id,
    is_local_path,
    infer_arch,
    resolve_model_path,
    _sanitise_repo_id,
)


# ---------------------------------------------------------------------------
# is_hf_repo_id
# ---------------------------------------------------------------------------

class TestIsHfRepoId:
    def test_simple_repo_id(self):
        assert is_hf_repo_id("stabilityai/sdxl-turbo") is True

    def test_dashed_repo_id(self):
        assert is_hf_repo_id("stable-diffusion-v1-5/stable-diffusion-v1-5") is True

    def test_local_path_not_repo(self):
        assert is_hf_repo_id("./models/sdxl-base-fp16") is False

    def test_absolute_path_not_repo(self):
        assert is_hf_repo_id("/home/user/models/foo") is False

    def test_existing_dir_not_repo(self, tmp_path):
        d = tmp_path / "foo/bar"
        d.mkdir(parents=True)
        # Even though it has a slash, it's an existing path → not a repo.
        assert is_hf_repo_id(str(d)) is False

    def test_empty_string(self):
        assert is_hf_repo_id("") is False

    def test_none(self):
        assert is_hf_repo_id(None) is False  # type: ignore[arg-type]

    def test_too_many_slashes(self):
        assert is_hf_repo_id("a/b/c") is False

    def test_home_path(self):
        assert is_hf_repo_id("~/models/foo") is False


# ---------------------------------------------------------------------------
# is_local_path
# ---------------------------------------------------------------------------

class TestIsLocalPath:
    def test_existing_dir(self, tmp_path):
        assert is_local_path(str(tmp_path)) is True

    def test_nonexistent(self):
        assert is_local_path("/nonexistent/path/xyz") is False

    def test_empty(self):
        assert is_local_path("") is False


# ---------------------------------------------------------------------------
# _sanitise_repo_id
# ---------------------------------------------------------------------------

class TestSanitiseRepoId:
    def test_basic(self):
        assert _sanitise_repo_id("stabilityai/sdxl-turbo") == "stabilityai--sdxl-turbo"

    def test_no_org(self):
        assert _sanitise_repo_id("foo/bar") == "foo--bar"


# ---------------------------------------------------------------------------
# resolve_model_path (local paths only; network tested separately)
# ---------------------------------------------------------------------------

class TestResolveModelPath:
    def test_existing_local_path(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        resolved = resolve_model_path(str(model_dir))
        assert resolved == os.path.abspath(str(model_dir))

    def test_passes_through_unknown(self):
        # A non-existent, non-repo-ID string is passed through to diffusers.
        result = resolve_model_path("just-a-name")
        assert result == "just-a-name"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resolve_model_path("")


# ---------------------------------------------------------------------------
# infer_arch
# ---------------------------------------------------------------------------

class TestInferArch:
    def test_sdxl_from_model_index(self, tmp_path):
        idx = {"_class_name": "StableDiffusionXLPipeline"}
        (tmp_path / "model_index.json").write_text(json.dumps(idx))
        assert infer_arch(str(tmp_path)) == "sdxl"

    def test_sd15_from_model_index(self, tmp_path):
        idx = {"_class_name": "StableDiffusionPipeline"}
        (tmp_path / "model_index.json").write_text(json.dumps(idx))
        assert infer_arch(str(tmp_path)) == "sd15"

    def test_fallback_sdxl(self, tmp_path):
        # No model_index.json, no sd15 keyword → defaults to sdxl.
        assert infer_arch(str(tmp_path)) == "sdxl"

    def test_sd15_keyword_in_name(self, tmp_path):
        d = tmp_path / "my-sd15-model"
        d.mkdir()
        assert infer_arch(str(d)) == "sd15"

    def test_sdxl_keyword_in_name(self, tmp_path):
        d = tmp_path / "sdxl-turbo-finetune"
        d.mkdir()
        assert infer_arch(str(d)) == "sdxl"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))