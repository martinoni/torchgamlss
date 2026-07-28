from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_public_project_community_files_are_present():
    expected = (
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/dependabot.yml",
        ".github/workflows/release.yml",
    )

    assert all((PROJECT_ROOT / path).is_file() for path in expected)


def test_public_readme_describes_status_scope_and_support():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "alpha software" in readme
    assert "not affiliated with or endorsed" in readme
    assert "gamlss-python" in readme
    assert "CONTRIBUTING.md" in readme
    assert "SECURITY.md" in readme
    assert "CITATION.cff" in readme
    assert "private pre-release" not in readme
    assert "private GitHub workflow artifacts" not in readme


def test_package_metadata_identifies_public_project_links():
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert project.count('{ name = "Thiago Martinoni" }') == 2
    assert 'Homepage = "https://github.com/martinoni/torchgamlss"' in project
    assert 'Repository = "https://github.com/martinoni/torchgamlss"' in project
    assert (
        'Documentation = "https://github.com/martinoni/torchgamlss/tree/main/docs"'
        in project
    )
    assert (
        'Changelog = '
        '"https://github.com/martinoni/torchgamlss/blob/main/CHANGELOG.md"'
        in project
    )


def test_citation_metadata_has_public_identity_and_license():
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert "family-names: Martinoni" in citation
    assert "given-names: Thiago" in citation
    assert "repository-code: https://github.com/martinoni/torchgamlss" in citation
    assert "license: GPL-3.0-only" in citation


def test_release_workflow_uses_testpypi_trusted_publishing():
    workflow = (
        PROJECT_ROOT / ".github/workflows/release.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "name: testpypi" in workflow
    assert "id-token: write" in workflow
    assert "https://test.pypi.org/legacy/" in workflow
    assert "pypa/gh-action-pypi-publish@" in workflow
    assert "secrets." not in workflow
    assert "password:" not in workflow
