import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from ravi_varma.config import GenerationConfig, GenerationParamsConfig
from ravi_varma.generation.controlnet import make_canny_conditioning


class TestConfigurationParsing:
    def test_width_height_must_be_multiple_of_8(self):
        with pytest.raises(ValidationError):
            GenerationParamsConfig(width=511)

    def test_defaults_load_without_a_file(self):
        config = GenerationConfig()
        assert config.generation.steps == 30
        assert config.style.trigger_token == "<rvvarma>"

    def test_load_from_dict_overrides_defaults(self):
        config = GenerationConfig.load({"generation": {"steps": 45}, "controlnet": {"enabled": True, "type": "canny"}})
        assert config.generation.steps == 45
        assert config.controlnet.enabled is True
        assert config.controlnet.type == "canny"


class TestCannyConditioning:
    def test_canny_preserves_image_size(self):
        img = Image.new("RGB", (256, 128), (50, 100, 150))
        edges = make_canny_conditioning(img)
        assert edges.size == (256, 128)

    def test_canny_output_is_grayscale_like_rgb(self):
        img = Image.new("RGB", (64, 64), (200, 30, 30))
        edges = make_canny_conditioning(img)
        pixels = list(edges.getdata())
        # every pixel should have R == G == B since it's a stacked edge mask
        assert all(p[0] == p[1] == p[2] for p in pixels[:100])


# --------------------------------------------------------------------------- #
# Mocked-pipeline tests: torch/diffusers are not installed in this sandbox
# (no GPU, no network access to the model hub), so we inject minimal fake
# modules that mimic just enough of their API surface for
# RaviVarmaGenerator.generate() to run end-to-end, and verify seed handling +
# metadata construction -- exactly as the project spec requires ("use
# mocked/lightweight pipelines so tests do not require downloading huge
# models").
# --------------------------------------------------------------------------- #
class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakeMpsBackend:
    @staticmethod
    def is_available():
        return False


class _FakeBackends:
    mps = _FakeMpsBackend()


class _FakeTorch(types.ModuleType):
    float16 = "float16"
    float32 = "float32"
    backends = _FakeBackends()

    class cuda:
        @staticmethod
        def is_available():
            return False

    Generator = _FakeGenerator


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakeScheduler:
    def __init__(self):
        self.config = {}

    @classmethod
    def from_config(cls, config):
        return cls()


class _FakePipeline:
    last_call_kwargs = None

    def __init__(self):
        self.scheduler = _FakeScheduler()
        self._lora_loaded_from = None

    def to(self, device):
        return self

    def enable_attention_slicing(self):
        pass

    def enable_vae_slicing(self):
        pass

    def enable_vae_tiling(self):
        pass

    def enable_model_cpu_offload(self):
        pass

    def load_lora_weights(self, path):
        self._lora_loaded_from = path

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def __call__(self, **kwargs):
        _FakePipeline.last_call_kwargs = kwargs
        return _FakeResult(Image.new("RGB", (kwargs.get("width", 512), kwargs.get("height", 512)), (10, 20, 30)))


@pytest.fixture
def fake_ml_stack(monkeypatch, tmp_path):
    fake_torch = _FakeTorch("torch")
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.StableDiffusionPipeline = _FakePipeline
    fake_diffusers.DPMSolverMultistepScheduler = _FakeScheduler
    fake_diffusers.EulerAncestralDiscreteScheduler = _FakeScheduler
    fake_diffusers.DDIMScheduler = _FakeScheduler
    fake_diffusers.PNDMScheduler = _FakeScheduler

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    yield
    _FakePipeline.last_call_kwargs = None


@pytest.fixture
def generation_config(tmp_path):
    return GenerationConfig.load(
        {
            "model": {"lora_path": str(tmp_path / "no_such_checkpoint")},
            "output": {"generated_dir": str(tmp_path / "generated")},
        }
    )


class TestSeedHandling:
    def test_explicit_seed_is_recorded_in_metadata(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config)
        result = gen.generate(prompt="a test prompt", seed=1234, save=False)
        assert result.metadata["seed"] == 1234

    def test_random_seed_used_when_none_given(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config)
        result = gen.generate(prompt="a test prompt", seed=None, save=False)
        assert isinstance(result.metadata["seed"], int)


class TestMetadataGeneration:
    def test_metadata_contains_all_expected_keys(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config)
        result = gen.generate(prompt="a test prompt", seed=1, save=False)
        for key in ("prompt", "seed", "steps", "guidance_scale", "width", "height", "lora_loaded", "disclaimer"):
            assert key in result.metadata

    def test_lora_loaded_false_when_checkpoint_missing(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config, allow_missing_lora=True)
        result = gen.generate(prompt="a test prompt", save=False)
        assert result.metadata["lora_loaded"] is False

    def test_missing_lora_raises_when_not_allowed(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import LoraCheckpointMissingError, RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config, allow_missing_lora=False)
        with pytest.raises(LoraCheckpointMissingError):
            gen.load()

    def test_save_writes_image_and_metadata_files(self, fake_ml_stack, generation_config):
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config)
        result = gen.generate(prompt="a test prompt", seed=1, save=True)
        assert result.image_path.exists()
        assert result.metadata_path.exists()


class TestMlStackUnavailable:
    def test_clear_error_when_torch_and_diffusers_missing(self, generation_config, monkeypatch):
        # Ensure a clean slate: simulate the real sandbox where torch/diffusers
        # genuinely are not importable.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch", "diffusers"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from ravi_varma.generation.pipeline import RaviVarmaGenerator

        gen = RaviVarmaGenerator(generation_config)
        with pytest.raises(RuntimeError, match="not usable"):
            gen.load()
