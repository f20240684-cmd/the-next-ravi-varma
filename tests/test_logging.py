import logging

from ravi_varma.utils.logging import setup_logging


def _capture(logger_name, message, tmp_path):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    from ravi_varma.utils.logging import _RedactSecretsFilter

    handler.addFilter(_RedactSecretsFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.warning(message)
    return stream.getvalue()


class TestSecretRedaction:
    def test_ordinary_message_with_colon_is_not_redacted(self, tmp_path):
        # Regression test: an earlier version of the filter had an operator
        # precedence bug ("A and B or C") that redacted ANY message
        # containing a colon, regardless of whether it had a secret in it.
        out = _capture("t1", "torch is not usable (OSError: missing shared library); falling back", tmp_path)
        assert "redacted" not in out
        assert "OSError" in out

    def test_message_with_token_and_equals_is_redacted(self, tmp_path):
        out = _capture("t2", "HF_TOKEN=abc123secret should be redacted", tmp_path)
        assert "redacted" in out
        assert "abc123secret" not in out

    def test_message_with_api_key_and_colon_is_redacted(self, tmp_path):
        out = _capture("t3", "api_key: sk-abcdef123 should be redacted", tmp_path)
        assert "redacted" in out
        assert "sk-abcdef123" not in out

    def test_setup_logging_is_idempotent(self):
        logger1 = setup_logging("idempotent_test", log_dir=None)
        logger2 = setup_logging("idempotent_test", log_dir=None)
        assert logger1 is logger2
        assert len(logger1.handlers) == 1
