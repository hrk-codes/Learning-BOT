from pathlib import Path
from tempfile import TemporaryDirectory

from memory.chat_memory import ChatMemory


def test_memory_lifecycle() -> None:
    with TemporaryDirectory() as temp_dir:
        history_path = Path(temp_dir) / "history.json"
        memory = ChatMemory(history_path=history_path, recent_message_limit=2)
        messages = []

        memory.add_message(messages, "user", "My name is Hrk.")
        memory.add_message(messages, "assistant", "Nice to meet you, Hrk.")
        memory.add_message(messages, "user", "What is my name?")
        memory.save_history(messages)

        loaded = memory.load_history()
        assert loaded.warning is None
        assert loaded.messages == messages

        context = memory.build_context("System prompt", loaded.messages)
        assert context == [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Nice to meet you, Hrk."},
            {"role": "user", "content": "What is my name?"},
        ]


def test_missing_corrupted_and_invalid_history() -> None:
    with TemporaryDirectory() as temp_dir:
        history_path = Path(temp_dir) / "history.json"
        memory = ChatMemory(history_path=history_path, recent_message_limit=10)

        missing = memory.load_history()
        assert missing.warning is None
        assert missing.messages == []

        history_path.write_text("{bad json", encoding="utf-8")
        corrupted = memory.load_history()
        assert corrupted.messages == []
        assert corrupted.warning is not None

        history_path.write_text(
            '[{"role": "user", "content": "valid"}, {"role": "system", "content": "skip me"}]',
            encoding="utf-8",
        )
        validated = memory.load_history()
        assert validated.messages == [{"role": "user", "content": "valid"}]
        assert validated.warning is not None


if __name__ == "__main__":
    test_memory_lifecycle()
    test_missing_corrupted_and_invalid_history()
    print("memory tests passed")
