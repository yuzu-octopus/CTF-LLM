from datasets import Dataset
from trl import SFTConfig


def test_sftconfig_accepts_assistant_only_loss():
    cfg = SFTConfig(output_dir="/tmp/x", assistant_only_loss=True)
    assert cfg.assistant_only_loss is True


def test_messages_dataset_roundtrip():
    ds = Dataset.from_list([{"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "ans"},
    ]}])
    assert "messages" in ds.column_names
    assert ds[0]["messages"][-1]["role"] == "assistant"
