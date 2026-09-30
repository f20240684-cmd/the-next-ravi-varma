import json

import pytest
from PIL import Image

from ravi_varma.config import DatasetConfig
from ravi_varma.data.dataset import prepare_dataset, load_manifest
from ravi_varma.data.validation import DatasetValidator


def _make_image(path, size=(600, 600), color=(120, 60, 30)):
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def dataset_layout(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_file = metadata_dir / "metadata.jsonl"

    _make_image(raw_dir / "0001.jpg")
    _make_image(raw_dir / "0002.jpg", color=(10, 10, 200))
    _make_image(raw_dir / "0003.jpg", size=(100, 100))  # below min resolution

    records = [
        {"image": "0001.jpg", "caption": "A royal woman in traditional attire", "artist": "Raja Ravi Varma"},
        {"image": "0002.jpg", "caption": "A mythological battlefield scene", "artist": "Raja Ravi Varma"},
        {"image": "0003.jpg", "caption": "A tiny low-res image", "artist": "Raja Ravi Varma"},
    ]
    with metadata_file.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    config = DatasetConfig(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        captions_dir=str(captions_dir),
        metadata_dir=str(metadata_dir),
        metadata_file=str(metadata_file),
        manifest_file=str(processed_dir / "manifest.jsonl"),
    )
    return config


class TestImageValidation:
    def test_flags_low_resolution(self, dataset_layout):
        report = DatasetValidator(dataset_layout).validate()
        low_res_issues = [i for i in report.issues if i.category == "low_resolution"]
        assert len(low_res_issues) == 1
        assert "0003.jpg" in low_res_issues[0].path

    def test_valid_samples_excludes_low_res(self, dataset_layout):
        report = DatasetValidator(dataset_layout).validate()
        assert report.valid_samples == 2
        assert report.is_valid

    def test_detects_corrupt_image(self, dataset_layout, tmp_path):
        corrupt_path = tmp_path / "raw" / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a real image")
        report = DatasetValidator(dataset_layout).validate()
        corrupt_issues = [i for i in report.issues if i.category == "corrupt_image"]
        assert len(corrupt_issues) == 1

    def test_empty_dataset_is_invalid(self, tmp_path):
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()
        config = DatasetConfig(raw_dir=str(raw_dir))
        report = DatasetValidator(config).validate()
        assert not report.is_valid
        assert any(i.category == "empty_dataset" for i in report.issues)


class TestMissingMetadata:
    def test_missing_metadata_file_warns_but_does_not_crash(self, tmp_path):
        raw_dir = tmp_path / "raw2"
        raw_dir.mkdir()
        _make_image(raw_dir / "img.jpg")
        config = DatasetConfig(raw_dir=str(raw_dir), metadata_file=str(tmp_path / "nope.jsonl"))
        report = DatasetValidator(config).validate()
        assert any(i.category == "no_metadata_file" for i in report.issues)

    def test_invalid_metadata_json_reported(self, tmp_path):
        raw_dir = tmp_path / "raw3"
        raw_dir.mkdir()
        _make_image(raw_dir / "img.jpg")
        bad_meta = tmp_path / "bad.jsonl"
        bad_meta.write_text("{not valid json}\n")
        config = DatasetConfig(raw_dir=str(raw_dir), metadata_file=str(bad_meta))
        report = DatasetValidator(config).validate()
        assert any(i.category == "invalid_metadata" for i in report.issues)


class TestCaptionLoading:
    def test_prepare_dataset_uses_metadata_captions(self, dataset_layout):
        manifest = prepare_dataset(dataset_layout)
        # 0003.jpg is below min resolution but prepare_dataset itself doesn't
        # re-run resolution checks -- it only requires a caption + readable
        # image, so it will still be included; validation is a separate gate.
        texts = {r.text for r in manifest}
        assert any("royal woman" in t for t in texts)
        assert all(t.startswith(dataset_layout.style.trigger_token) for t in texts)

    def test_prepare_dataset_skips_missing_caption(self, tmp_path):
        raw_dir = tmp_path / "raw4"
        raw_dir.mkdir()
        _make_image(raw_dir / "nocaption.jpg")
        config = DatasetConfig(raw_dir=str(raw_dir), processed_dir=str(tmp_path / "proc4"), manifest_file=str(tmp_path / "proc4" / "manifest.jsonl"))
        manifest = prepare_dataset(config)
        assert manifest == []


class TestManifestCreation:
    def test_manifest_round_trip(self, dataset_layout):
        manifest = prepare_dataset(dataset_layout)
        assert len(manifest) == 3
        loaded = load_manifest(dataset_layout.manifest_file)
        assert len(loaded) == len(manifest)
        assert loaded[0].text == manifest[0].text

    def test_load_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "does_not_exist.jsonl")
