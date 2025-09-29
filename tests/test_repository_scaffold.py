from pathlib import Path


def test_core_directories_present():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "core/domain",
        "core/application",
        "ports",
        "adapters/primary",
        "adapters/secondary",
        "config",
    ]:
        assert (root / rel).is_dir(), f"Expected directory '{rel}' to exist"


def test_plan_document_present():
    root = Path(__file__).resolve().parents[1]
    plan = root / "IMPLEMENTATION_PLAN.md"
    assert plan.is_file(), "Implementation plan must accompany repository"
    content = plan.read_text(encoding="utf-8")
    assert "AutoTS Hexagonal Redesign" in content
