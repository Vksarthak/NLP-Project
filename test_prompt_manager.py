"""Tests for PromptManager."""

import os
import tempfile

import pytest

from src.prompt_manager import PromptManager, PromptError


@pytest.fixture
def prompt_dir():
    """Create a temp directory with sample .txt prompt templates."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Simple template with two variables.
        with open(os.path.join(tmp_dir, "greeting.txt"), "w") as f:
            f.write("Hello {name}, welcome to {place}.")

        # Template with escaped braces (like our real prompts for JSON output).
        with open(os.path.join(tmp_dir, "json_output.txt"), "w") as f:
            f.write("Analyze: {document_text}\nOutput: {{\"result\": \"ok\"}}")

        yield tmp_dir


# ------------------------------------------------------------------
# Template loading
# ------------------------------------------------------------------

class TestLoadTemplate:
    def test_loads_existing_template(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        content = mgr.load_template("greeting")
        assert "{name}" in content
        assert "{place}" in content

    def test_auto_appends_txt_extension(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        content = mgr.load_template("greeting")  # no .txt
        assert "Hello" in content

    def test_loads_with_explicit_extension(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        content = mgr.load_template("greeting.txt")
        assert "Hello" in content

    def test_missing_template_raises(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        with pytest.raises(PromptError, match="not found"):
            mgr.load_template("nonexistent")


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

class TestRender:
    def test_renders_with_variables(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        result = mgr.render("greeting", name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland."

    def test_missing_variable_raises(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        with pytest.raises(PromptError, match="Missing required variable"):
            mgr.render("greeting", name="Alice")  # missing 'place'

    def test_escaped_braces_preserved(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        result = mgr.render("json_output", document_text="some text")
        assert '{"result": "ok"}' in result
        assert "some text" in result


# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------

class TestCaching:
    def test_template_is_cached(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        mgr.load_template("greeting")
        assert "greeting.txt" in mgr.cache

    def test_cached_version_used_after_file_change(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        # First load — populates cache.
        mgr.render("greeting", name="A", place="B")

        # Modify the file on disk.
        with open(os.path.join(prompt_dir, "greeting.txt"), "w") as f:
            f.write("Changed {name}")

        # Should still use the cached (original) version.
        result = mgr.render("greeting", name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland."


# ------------------------------------------------------------------
# Listing
# ------------------------------------------------------------------

class TestListTemplates:
    def test_lists_all_templates(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        templates = mgr.list_templates()
        assert "greeting" in templates
        assert "json_output" in templates
        assert len(templates) == 2

    def test_returns_sorted(self, prompt_dir):
        mgr = PromptManager(prompts_dir=prompt_dir)
        templates = mgr.list_templates()
        assert templates == sorted(templates)
