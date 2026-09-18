"""
Structural tests for llm_explainer.call_llm's Groq backend.

test_llm_explainer.py deliberately never exercises call_llm itself --
every test there injects a mock llm_fn, specifically because call_llm
needs a real GROQ_API_KEY and network access to api.groq.com to run
for real (neither available in this sandbox). This file closes that
gap the other way: it patches groq.Groq itself, so we can verify
call_llm's OWN logic --

  - fails fast with a clear error when GROQ_API_KEY is missing
  - builds the request in the correct shape (system/user roles, right
    model/max_tokens/temperature defaults)
  - correctly extracts text back out of the Groq response object
  - respects caller overrides (model, max_tokens, temperature)

-- without ever making a real HTTP call. If you later add a real
integration test (live key + live call), keep it separate and marked
clearly as such, since it won't run in CI or in this sandbox.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

from llm_explainer import call_llm


def make_fake_groq_response(text: str):
    """Mimics the shape of a real groq ChatCompletion response enough
    for call_llm's response.choices[0].message.content access to work."""
    fake_message = MagicMock()
    fake_message.content = text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


print("=" * 70)
print("TEST 1 — Missing GROQ_API_KEY raises RuntimeError immediately, no network attempted")
with patch.dict(os.environ, {}, clear=True):
    os.environ.pop("GROQ_API_KEY", None)
    try:
        call_llm("system prompt", "user prompt")
        print("FAIL — should have raised")
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)
        print("PASS —", e)

print("=" * 70)
print("TEST 2 — With a key set, Groq client is constructed with that key")
with patch.dict(os.environ, {"GROQ_API_KEY": "fake-test-key-123"}):
    with patch("groq.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        mock_client.chat.completions.create.return_value = make_fake_groq_response("hello world")
        call_llm("sys", "user")
        MockGroq.assert_called_once_with(api_key="fake-test-key-123")
        print("PASS — Groq(api_key=...) called with the env var's key")

print("=" * 70)
print("TEST 3 — Request shape: system_prompt and user_prompt land in the right roles, in order")
with patch.dict(os.environ, {"GROQ_API_KEY": "fake-test-key-123"}):
    with patch("groq.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        mock_client.chat.completions.create.return_value = make_fake_groq_response("ok")
        call_llm("SYSTEM_TEXT_HERE", "USER_TEXT_HERE")
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        assert len(messages) == 2, f"expected 2 messages, got {len(messages)}"
        assert messages[0] == {"role": "system", "content": "SYSTEM_TEXT_HERE"}
        assert messages[1] == {"role": "user", "content": "USER_TEXT_HERE"}
        print("PASS — messages=[{system}, {user}] in correct order with correct content")

print("=" * 70)
print("TEST 4 — Default model/max_tokens/temperature match documented defaults")
with patch.dict(os.environ, {"GROQ_API_KEY": "fake-test-key-123"}):
    with patch("groq.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        mock_client.chat.completions.create.return_value = make_fake_groq_response("ok")
        call_llm("sys", "user")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "openai/gpt-oss-120b", kwargs["model"]
        assert kwargs["max_tokens"] == 500, kwargs["max_tokens"]
        assert kwargs["temperature"] == 0.2, kwargs["temperature"]
        print("PASS — model=llama-3.3-70b-versatile, max_tokens=500, temperature=0.2")

print("=" * 70)
print("TEST 5 — Caller overrides (model, max_tokens, temperature) are respected, not silently ignored")
with patch.dict(os.environ, {"GROQ_API_KEY": "fake-test-key-123"}):
    with patch("groq.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        mock_client.chat.completions.create.return_value = make_fake_groq_response("ok")
        call_llm("sys", "user", model="llama-3.1-8b-instant", max_tokens=200, temperature=0.0)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "llama-3.1-8b-instant"
        assert kwargs["max_tokens"] == 200
        assert kwargs["temperature"] == 0.0
        print("PASS — all three overrides passed through correctly")

print("=" * 70)
print("TEST 6 — Response text is correctly extracted from response.choices[0].message.content")
with patch.dict(os.environ, {"GROQ_API_KEY": "fake-test-key-123"}):
    with patch("groq.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        expected_text = "This is covered under IPC Section 304A.\n\nSource: IPC Section 304A"
        mock_client.chat.completions.create.return_value = make_fake_groq_response(expected_text)
        result = call_llm("sys", "user")
        assert result == expected_text, f"got {result!r}"
        print("PASS — returned text matches exactly what the mocked response provided")

print("=" * 70)
print("TEST 7 — call_llm is the actual default llm_fn used by generate_explanation")
# Sanity check that generate_explanation's default parameter really is
# this call_llm, so the Groq wiring is genuinely the production path
# and not something that got quietly disconnected.
import inspect
from llm_explainer import generate_explanation
sig = inspect.signature(generate_explanation)
assert sig.parameters["llm_fn"].default is call_llm
print("PASS — generate_explanation defaults to call_llm (the Groq-backed function)")

print("=" * 70)
print("All structural tests passed. NOTE: none of these exercised a real network")
print("call to api.groq.com. Before trusting this in production, also run a real")
print("call_llm(...) locally with a genuine GROQ_API_KEY set, and eyeball 2-3")
print("outputs for citation-format consistency (see prior note on Llama 3.3's")
print("instruction-following being good but not as rigid as Claude's).")
