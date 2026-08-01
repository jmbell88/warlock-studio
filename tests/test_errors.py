from __future__ import annotations

import httpx

from animancer3d.errors import friendly, write_error_log


def test_oom_message_is_friendly():
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert "resolution 512" in friendly(exc)


def test_transport_error_points_at_the_log():
    exc = httpx.TransportError("connection reset")
    assert "trellis.log" in friendly(exc)


def test_unknown_exception_falls_back_to_str():
    assert friendly(ValueError("weird")) == "weird"


def test_exception_with_no_message_falls_back_to_class_name():
    assert friendly(RuntimeError()) == "RuntimeError"


def test_write_error_log_captures_full_traceback(tmp_path):
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        write_error_log(tmp_path / "job1", exc)
    content = (tmp_path / "job1" / "error.log").read_text()
    assert "RuntimeError: boom" in content
    assert "Traceback" in content
