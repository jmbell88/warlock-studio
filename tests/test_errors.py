from __future__ import annotations

import errno
import subprocess

import httpx

from warlock.errors import MAX_MESSAGE, friendly, write_error_log


def test_a_multi_line_exception_keeps_only_its_first_line():
    """The fallback used to hand ``str(exc)`` straight to the DB and the
    inspector. A subprocess error's str is routinely its child's whole stderr,
    which is exactly the traceback fragment tasks.py takes care never to
    toast."""
    exc = ValueError("the mesh could not be read\nTraceback (most recent call last):\n  File x")
    assert friendly(exc) == "the mesh could not be read"


def test_a_long_message_is_bounded():
    exc = ValueError("boom " * 500)
    message = friendly(exc)
    assert len(message) <= MAX_MESSAGE
    assert message.endswith("...")


def test_a_full_disk_names_a_remedy_rather_than_an_errno():
    exc = OSError(errno.ENOSPC, "No space left on device")
    message = friendly(exc)
    assert "Errno" not in message
    assert "space" in message.lower()
    assert "prune" in message.lower()


def test_a_held_file_names_the_path_and_the_cause():
    exc = PermissionError(errno.EACCES, "Permission denied", "model.glb")
    message = friendly(exc)
    assert "model.glb" in message
    assert "another program" in message.lower()


def test_a_failed_helper_program_points_at_the_log():
    exc = subprocess.CalledProcessError(3, ["blender", "--background"])
    message = friendly(exc)
    assert "blender" in message
    assert "warlock.log" in message


def test_host_memory_exhaustion_is_not_reported_as_a_gpu_problem():
    """MemoryError is the host running out of commit, which is the 2026-08-03
    resource-exhaustion crash -- telling that user to drop the resolution to
    512 sends them after the wrong resource."""
    message = friendly(MemoryError())
    assert "512" not in message
    assert "system memory" in message.lower()


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
