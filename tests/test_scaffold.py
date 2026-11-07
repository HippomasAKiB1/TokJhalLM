"""Unit tests for TokJhalLM repository scaffold, configurations, and environment checks."""

from pathlib import Path
import pytest
import yaml

from scripts.smoke_test import REQUIRED_CONFIG_KEYS, validate_config


@pytest.fixture
def repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


def test_base_yaml_exists_and_valid(repo_root: Path) -> None:
    """Verify configs/base.yaml exists and satisfies schema requirements."""
    base_yaml_path = repo_root / "configs" / "base.yaml"
    assert base_yaml_path.is_file(), "configs/base.yaml does not exist"

    assert "model" in REQUIRED_CONFIG_KEYS
    config = validate_config(base_yaml_path)

    # Validate model parameters per PRD §6.1
    model = config["model"]
    assert model["layers"] == 8
    assert model["hidden"] == 640
    assert model["heads"] == 8
    assert model["vocab_size"] == 12000
    assert model["context_len"] == 128
    assert model["tied_embeddings"] is True
    assert model["activation"] == "swiglu"
    assert model["norm"] == "rmsnorm"
    assert model["pos_enc"] == "rope"
    assert model["dropout"] == 0.1

    # Validate training hyperparameters per PRD §6.3
    training = config["training"]
    assert training["batch_size"] == 2
    assert training["grad_accum"] == 16
    assert training["lr"] == 5.0e-4
    assert training["warmup_steps"] == 500
    assert training["weight_decay"] == 0.01
    assert training["optimizer"] == "adamw_8bit"
    assert training["grad_clip"] == 1.0
    assert training["precision"] in ["bf16", "fp16"]
    assert training["gradient_checkpointing"] is True
    assert training["seed"] == 42


def test_phase0_yaml_structure(repo_root: Path) -> None:
    """Verify configs/phase0.yaml exists and adheres to PRD §5.0 sequential structure."""
    phase0_yaml_path = repo_root / "configs" / "phase0.yaml"
    assert phase0_yaml_path.is_file(), "configs/phase0.yaml does not exist"

    with phase0_yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "f1_floor" in config
    assert config["f1_floor"] == 0.40

    runs = config.get("runs", {})
    assert "stage_a" in runs
    assert "stage_b" in runs
    assert "stage_c" in runs

    # Stage a has 2 runs
    stage_a = runs["stage_a"]
    assert "run_1_mlm_50m_bpe" in stage_a
    assert "run_2_electra_50m_bpe" in stage_a

    # Stage b has 1 new run + reuse reference
    stage_b = runs["stage_b"]
    assert "run_3_winning_30m_bpe" in stage_b
    assert "reused_from_stage_a" in stage_b

    # Stage c has 1 new run with TBD placeholders + reuse reference
    stage_c = runs["stage_c"]
    assert "run_4_winning_obj_size_unigram" in stage_c
    assert "reused_from_stage_b" in stage_c
    assert stage_c["run_4_winning_obj_size_unigram"]["task"] == "TBD"
    assert stage_c["run_4_winning_obj_size_unigram"]["size"] == "TBD"


def test_readme_contains_tagline(repo_root: Path) -> None:
    """Verify README.md contains verbatim tagline and hardware envelope."""
    readme_path = repo_root / "README.md"
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")

    assert "TokJhalLM" in content
    assert "BanglaBERT is bland. TokJhalLM has flavor." in content
    assert "4GB VRAM · 8GB RAM · i5-12th gen CPU" in content


def test_gpu_forward_backward_if_cuda_available() -> None:
    """Test GPU forward and backward pass, skipping if CUDA is not available."""
    import torch
    from scripts.smoke_test import verify_cuda, verify_precision_forward_backward

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    device_name, vram_gb, bf16_supported = verify_cuda()
    assert vram_gb > 0
    precision_used = verify_precision_forward_backward(bf16_supported)
    assert precision_used in ["bf16", "fp16"]
