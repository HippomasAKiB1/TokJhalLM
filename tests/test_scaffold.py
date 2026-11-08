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

    # Validate model parameters per PRD §6.1; default size is 30M per D15
    model = config["model"]
    assert model["layers"] == 8
    assert model["hidden"] == 480
    assert model["heads"] == 6
    assert model["hidden"] // model["heads"] == 80
    assert model["ffn_hidden"] == 1440
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
    # GTX 1650 has no bf16 support (Hardware truths in CLAUDE.md); fp32 allowed pending decision D1.
    assert training["precision"] in ["fp16", "fp32"]
    assert training["gradient_checkpointing"] is True
    assert training["seed"] == 42


def test_phase0_yaml_structure(repo_root: Path) -> None:
    """Verify configs/phase0.yaml follows PRD §5.0 as amended by D15 (3 runs, all 30M)."""
    phase0_yaml_path = repo_root / "configs" / "phase0.yaml"
    assert phase0_yaml_path.is_file(), "configs/phase0.yaml does not exist"

    with phase0_yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "f1_floor" in config
    assert config["f1_floor"] == 0.40

    runs = config.get("runs", {})
    assert set(runs) == {"stage_a", "stage_b"}

    # Stage a: MLM vs ELECTRA at 30M / BPE
    stage_a = runs["stage_a"]
    assert stage_a["run_1_mlm_30m_bpe"]["objective"] == "mlm"
    assert stage_a["run_2_electra_30m_bpe"]["objective"] == "electra"

    # Stage b: BPE vs Unigram at the winning objective, BPE run reused from stage a
    stage_b = runs["stage_b"]
    assert "reused_from_stage_a" in stage_b
    assert stage_b["run_3_winning_obj_30m_unigram"]["objective"] == "TBD"
    assert stage_b["run_3_winning_obj_30m_unigram"]["tokenizer"] == "unigram"

    # Every stage uses the same run-spec keys, and every run is 30M (D15)
    run_specs = [
        spec
        for stage in runs.values()
        for name, spec in stage.items()
        if name.startswith("run_")
    ]
    assert len(run_specs) == 3
    assert all(spec["params"] == "30M" for spec in run_specs)
    for spec in run_specs:
        assert set(spec) == {"objective", "params", "tokenizer", "vocab_size"}


def _estimate_encoder_params(
    layers: int, hidden: int, ffn_hidden: int, vocab_size: int
) -> int:
    """Estimate encoder params: tied embeddings, bias-free attention + SwiGLU, RMSNorm,
    and an MLM/RTD head transform (hidden×hidden + bias + norm)."""
    embeddings = vocab_size * hidden
    per_layer = 4 * hidden * hidden + 3 * hidden * ffn_hidden + 2 * hidden
    head = hidden * hidden + 2 * hidden
    return embeddings + layers * per_layer + hidden + head


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)
    return data


def test_phase0_model_sizes_hit_targets(repo_root: Path) -> None:
    """Verify each Phase 0 size label maps to a model config near its nominal size."""
    config = _load_yaml(repo_root / "configs" / "phase0.yaml")
    sizes = config["model_sizes"]
    assert sizes == {"30M": "configs/model_30m.yaml"}
    assert config["not_selected"] == {"50M": "configs/model_50m.yaml"}  # D15: reference only

    # Every `params` label used by a concrete run must be defined
    for stage in config["runs"].values():
        for name, spec in stage.items():
            if name.startswith("run_") and spec["params"] != "TBD":
                assert spec["params"] in sizes

    for label, rel_path in {**sizes, **config["not_selected"]}.items():
        dims = _load_yaml(repo_root / rel_path)["model"]
        nominal_m = float(label.rstrip("M"))
        assert dims["layers"] == 8, "depth held fixed per PRD §6.1"
        assert dims["hidden"] // dims["heads"] == 80, "head_dim held fixed per PRD §6.1"
        est_m = _estimate_encoder_params(
            dims["layers"], dims["hidden"], dims["ffn_hidden"], dims["vocab_size"]
        ) / 1e6
        assert abs(est_m - nominal_m) / nominal_m < 0.05, f"{label}: estimated {est_m:.1f}M"
        assert abs(est_m - dims["est_params_m"]) < 0.1


def test_base_model_matches_model_30m(repo_root: Path) -> None:
    """base.yaml model section must mirror configs/model_30m.yaml (the selected size, D15)."""
    base = _load_yaml(repo_root / "configs" / "base.yaml")["model"]
    model_30m = _load_yaml(repo_root / "configs" / "model_30m.yaml")["model"]
    for key, value in base.items():
        assert model_30m[key] == value, key


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
    assert precision_used == "fp16"  # GTX 1650: no bf16; fp16 is the tested fallback path
