from warlock.studio.fps import FpsMeter


def test_an_empty_meter_reports_zero_rather_than_dividing_by_nothing():
    meter = FpsMeter()
    assert meter.fps == 0.0
    assert meter.frame_ms == 0.0
    assert meter.worst_ms == 0.0
    assert meter.average_fps == 0.0


def test_a_steady_sixty_hertz_stream_reads_as_sixty():
    meter = FpsMeter()
    for _ in range(120):
        meter.record(1 / 60)
    assert round(meter.fps, 3) == 60.0
    assert round(meter.frame_ms, 2) == 16.67
    assert round(meter.average_fps, 3) == 60.0


def test_the_window_forgets_but_the_session_total_does_not():
    meter = FpsMeter(window=10)
    for _ in range(10):
        meter.record(1 / 30)
    for _ in range(10):
        meter.record(1 / 60)
    # The window holds only the recent half...
    assert round(meter.fps, 1) == 60.0
    # ...while the session average spans both halves: 20 frames in 0.5 s.
    assert meter.frames == 20
    assert round(meter.average_fps, 1) == 40.0


def test_one_stall_shows_in_the_peak_where_the_mean_would_hide_it():
    meter = FpsMeter()
    for _ in range(119):
        meter.record(1 / 60)
    meter.record(0.1)
    assert meter.fps > 55.0  # the mean barely moves
    assert round(meter.worst_ms, 1) == 100.0  # the peak does not


def test_a_clock_that_did_not_advance_is_not_a_free_frame():
    meter = FpsMeter()
    meter.record(1 / 60)
    meter.record(0.0)
    meter.record(-0.5)
    assert meter.frames == 1
    assert round(meter.fps, 1) == 60.0
