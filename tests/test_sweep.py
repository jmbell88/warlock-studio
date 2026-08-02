"""The band sweep itself needs a GPU and the vendored exe; what is testable
here is the argument handling and the reporting, which is where a mistake would
quietly waste several minutes of GPU time per band."""

from __future__ import annotations

import asyncio

import pytest

from warlock import sweep
from warlock.config import Config


def test_auto_becomes_none_so_the_flag_is_omitted():
    # None is load-bearing: TrellisServer omits --band entirely rather than
    # passing "auto", which the exe has no keyword for.
    assert sweep.parse_bands("auto") == [None]
    assert sweep.parse_bands("none") == [None]
    assert sweep.parse_bands("default") == [None]


def test_parses_a_ladder_preserving_order():
    assert sweep.parse_bands("auto,2,4,8") == [None, 2, 4, 8]


def test_duplicates_are_dropped():
    # Each duplicate would be a full generation for an answer already known.
    assert sweep.parse_bands("4,auto,4,auto,2") == [4, None, 2]


def test_whitespace_and_empty_entries_are_tolerated():
    assert sweep.parse_bands(" 2 , ,4 ") == [2, 4]


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        sweep.parse_bands(",,")


def test_non_numeric_band_is_an_error():
    with pytest.raises(ValueError):
        sweep.parse_bands("wide")


def test_default_ladder_includes_the_baseline():
    assert sweep.parse_bands(sweep.DEFAULT_BANDS)[0] is None


def test_a_clear_win_over_auto_is_recommended(capsys):
    rows = [
        {"band": "auto", "worst": 0.31, "mean": 0.19, "faces": 300_000, "seconds": 100.0},
        {"band": "4", "worst": 0.02, "mean": 0.01, "faces": 310_000, "seconds": 110.0},
        {"band": "8", "worst": 0.08, "mean": 0.04, "faces": 320_000, "seconds": 120.0},
    ]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "lowest: band 4" in out
    assert "DEFAULT_TRELLIS_BAND = 4" in out


def test_a_win_inside_the_noise_is_not_recommended(capsys):
    """The real 2026-08-01 sweep, verbatim. At res 1024 the exe's res/512
    heuristic *is* band 2, so 'auto' and '2' are the same setting run twice --
    and band 2 'won' by less than the 4th decimal place. Recommending a
    hard-coded value off that would be tuning on noise."""
    rows = [
        {"band": "auto", "worst": 0.00771, "mean": 0.0023, "faces": 267_360, "seconds": 123.0},
        {"band": "2", "worst": 0.00770, "mean": 0.0023, "faces": 266_632, "seconds": 142.9},
        {"band": "4", "worst": 0.0167, "mean": 0.0078, "faces": 290_774, "seconds": 124.4},
        {"band": "8", "worst": 0.0110, "mean": 0.0048, "faces": 297_898, "seconds": 136.0},
        {"band": "16", "worst": 0.0125, "mean": 0.0033, "faces": 289_586, "seconds": 193.3},
    ]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "inside the noise" in out
    assert "Leave DEFAULT_TRELLIS_BAND = None" in out
    assert "DEFAULT_TRELLIS_BAND = 2" not in out


def test_auto_winning_outright_says_leave_it_alone(capsys):
    rows = [
        {"band": "auto", "worst": 0.007, "mean": 0.002, "faces": 267_000, "seconds": 123.0},
        {"band": "4", "worst": 0.017, "mean": 0.008, "faces": 290_000, "seconds": 124.4},
    ]
    sweep.print_table(rows, 1024)
    assert "leave DEFAULT_TRELLIS_BAND = None" in capsys.readouterr().out


def test_a_failed_band_does_not_hide_the_others(capsys):
    rows = [
        {"band": "auto", "worst": 0.31, "mean": 0.19, "faces": 300_000, "seconds": 100.0},
        {"band": "4", "worst": 0.02, "mean": 0.01, "faces": 310_000, "seconds": 110.0},
        {"band": "8", "error": "trellis-server exited during startup"},
    ]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "failed" in out
    assert "lowest: band 4" in out


def test_a_missing_baseline_blocks_any_recommendation(capsys):
    """Without 'auto' there is nothing to judge a margin against."""
    rows = [{"band": "4", "worst": 0.02, "mean": 0.01, "faces": 310_000, "seconds": 110.0}]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "no 'auto' row" in out
    assert "DEFAULT_TRELLIS_BAND = 4" not in out


def test_table_survives_every_band_failing(capsys):
    sweep.print_table([{"band": "auto", "error": "boom"}], 1024)
    assert "every band failed" in capsys.readouterr().out


# --- port ownership ---
#
# ensure_started() only polls /health; it cannot tell its own subprocess from
# someone else's. If the app is already listening, every band would be measured
# against that server and the sweep would print a confident table of identical,
# wrong numbers. Observed for real: a sweep-shaped run attached to a
# pre-existing server 30 ms after "starting trellis-server".


def test_occupied_port_aborts_the_sweep(tmp_path):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        config = Config(
            data_dir=tmp_path,
            db_path=tmp_path / "jobs.sqlite",
            trellis_port=port,
        )
        with pytest.raises(SystemExit, match="port"):
            sweep.require_free_port(config)


def test_free_port_is_accepted(tmp_path):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = Config(data_dir=tmp_path, db_path=tmp_path / "jobs.sqlite", trellis_port=port)
    sweep.require_free_port(config)  # must not raise


def test_missing_reference_image_aborts_before_any_gpu_work(tmp_path):
    config = Config(data_dir=tmp_path, db_path=tmp_path / "jobs.sqlite")
    with pytest.raises(SystemExit, match="not found"):
        asyncio.run(
            sweep.sweep(
                config,
                tmp_path / "nope.png",
                [None],
                tmp_path / "out",
                seed=1,
                resolution=512,
                audit_resolution=256,
            )
        )
