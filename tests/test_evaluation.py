import json
from pathlib import Path

import pytest
from PIL import Image

from ravi_varma.evaluation.clip_score import clip_available, text_image_similarity
from ravi_varma.evaluation.report import batch_evaluate, evaluate_image
from ravi_varma.evaluation.style_similarity import compute_style_similarity


def _make_image(path, color=(80, 40, 20)):
    Image.new("RGB", (64, 64), color).save(path)


class TestMetricCalculationsWithoutCLIP:
    """In this sandbox, open_clip/torch are not installed, so these tests
    exercise the real 'CLIP unavailable' code path -- verifying we return
    None rather than a fabricated score, per the project spec."""

    def test_clip_reports_unavailable(self):
        availability = clip_available()
        # This sandbox has no torch/open_clip installed; if a future
        # environment *does* have them, this simply confirms the flag is
        # a bool with a reason when False.
        assert isinstance(availability.available, bool)
        if not availability.available:
            assert availability.reason

    def test_text_image_similarity_returns_none_without_clip(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        _make_image(img_path)
        if clip_available().available:
            pytest.skip("CLIP is available in this environment; None-path not exercised.")
        assert text_image_similarity(img_path, "a painting") is None

    def test_style_similarity_returns_none_without_clip(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        _make_image(img_path)
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        _make_image(ref_dir / "ref1.jpg")
        if clip_available().available:
            pytest.skip("CLIP is available in this environment; None-path not exercised.")
        result = compute_style_similarity(img_path, ref_dir)
        assert result.mean_similarity is None
        assert result.num_references_used == 0


class TestMissingImages:
    def test_evaluate_image_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evaluate_image(tmp_path / "does_not_exist.png", tmp_path / "reference")

    def test_batch_evaluate_on_directory_with_no_images(self, tmp_path):
        empty_dir = tmp_path / "empty_generated"
        empty_dir.mkdir()
        results = batch_evaluate(empty_dir, tmp_path / "reference", tmp_path / "report.csv")
        assert results == []


class TestEmptyReferenceDirectory:
    def test_empty_reference_dir_returns_no_references_used(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        ref_dir = tmp_path / "empty_reference"
        ref_dir.mkdir()
        result = compute_style_similarity(img_path, ref_dir)
        assert result.num_references_used == 0
        assert result.mean_similarity is None

    def test_nonexistent_reference_dir_handled_gracefully(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        result = compute_style_similarity(img_path, tmp_path / "does_not_exist_dir")
        assert result.num_references_used == 0


class TestSidecarPromptLoading:
    def test_evaluate_image_reads_prompt_from_sidecar_json(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        sidecar = tmp_path / "gen.json"
        sidecar.write_text(json.dumps({"prompt": "a royal portrait"}))
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()

        result = evaluate_image(img_path, ref_dir)
        assert result.prompt == "a royal portrait"

    def test_evaluate_image_no_prompt_no_sidecar_sets_none(self, tmp_path):
        img_path = tmp_path / "gen2.png"
        _make_image(img_path)
        ref_dir = tmp_path / "reference2"
        ref_dir.mkdir()

        result = evaluate_image(img_path, ref_dir)
        assert result.prompt is None
        assert result.text_alignment is None
