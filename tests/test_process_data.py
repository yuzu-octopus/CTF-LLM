from src.process_data import convert_alpaca_to_chat, is_ctf_content


def test_convert_alpaca_to_chat_includes_system_by_default():
    out = convert_alpaca_to_chat("Solve picoCTF", "challenge desc", "ret2win", "pwn")
    assert out["messages"][0]["role"] == "system"


def test_convert_alpaca_to_chat_skip_system_prompt():
    out = convert_alpaca_to_chat("Solve picoCTF", "desc", "ret2win", "pwn", skip_system_prompt=True)
    assert len(out["messages"]) == 2
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][1]["role"] == "assistant"


def test_skip_flag_preserves_user_content():
    out = convert_alpaca_to_chat("Solve picoCTF", "desc", "ret2win", "pwn", skip_system_prompt=True)
    assert "Solve picoCTF" in out["messages"][0]["content"]
    assert "desc" in out["messages"][0]["content"]


def test_is_ctf_content_matches_categories():
    for w in ["pwn", "rev", "crypto", "web"]:
        assert is_ctf_content(w)


def test_is_ctf_content_rejects_unrelated():
    assert not is_ctf_content("kitchen")
