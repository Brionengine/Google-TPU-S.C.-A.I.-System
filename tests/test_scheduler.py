"""
Tests for TPU workload scheduling and capacity accounting.

The scheduler's job is to never promise the same chip twice, so most of these
assert on capacity bookkeeping across submit / complete cycles.
"""

import pytest

from framework import TPUAllocation, TPUWorkloadScheduler


@pytest.fixture
def allocations():
    return [
        TPUAllocation('zone-a', 'v6e', 64, 'spot'),
        TPUAllocation('zone-b', 'v4', 32, 'on-demand'),
    ]


@pytest.fixture
def scheduler(allocations):
    return TPUWorkloadScheduler(allocations)


# -- Capacity accounting ----------------------------------------------------


def test_scheduling_consumes_capacity(scheduler, allocations):
    scheduler.submit_job('train', required_chips=32, preferred_chip='v6e')

    assert scheduler.available_chips(allocations[0]) == 32


def test_capacity_is_not_oversubscribed(scheduler):
    """Regression: every job used to be scheduled against the *total* count."""
    first = scheduler.submit_job('train', 64, preferred_chip='v6e')
    second = scheduler.submit_job('train', 64, preferred_chip='v6e')

    assert first['status'] == 'scheduled'
    assert second['status'] == 'queued'


def test_total_committed_never_exceeds_fleet(scheduler):
    for _ in range(20):
        scheduler.submit_job('train', 16)

    report = scheduler.get_total_compute()
    assert report['chips_in_use'] <= report['total_chips']


def test_completion_releases_capacity(scheduler, allocations):
    job = scheduler.submit_job('train', 64, preferred_chip='v6e')
    scheduler.complete_job(job['id'])

    assert scheduler.available_chips(allocations[0]) == 64


def test_completion_lets_a_queued_job_run(scheduler):
    first = scheduler.submit_job('train', 64, preferred_chip='v6e')
    second = scheduler.submit_job('train', 64, preferred_chip='v6e')
    assert second['status'] == 'queued'

    scheduler.complete_job(first['id'])

    assert second['status'] == 'scheduled'
    assert second['id'] in scheduler.running_jobs


def test_queue_drains_in_submission_order(scheduler):
    running = scheduler.submit_job('train', 64, preferred_chip='v6e')
    a = scheduler.submit_job('a', 64, preferred_chip='v6e')
    b = scheduler.submit_job('b', 64, preferred_chip='v6e')

    scheduler.complete_job(running['id'])

    assert a['status'] == 'scheduled'
    assert b['status'] == 'queued'


def test_unplaceable_job_keeps_its_place_in_the_queue(scheduler):
    """A job too large for any zone must not block smaller ones behind it."""
    scheduler.submit_job('huge', 500)
    small = scheduler.submit_job('small', 8)

    assert small['status'] == 'scheduled'
    assert len(scheduler.job_queue) == 1


def test_job_larger_than_any_allocation_stays_queued(scheduler):
    job = scheduler.submit_job('huge', 1000)

    assert job['status'] == 'queued'
    assert job['assigned_zone'] is None


def test_completing_an_unknown_job_returns_none(scheduler):
    assert scheduler.complete_job(9999) is None


def test_completing_twice_does_not_double_release(scheduler, allocations):
    job = scheduler.submit_job('train', 32, preferred_chip='v6e')
    scheduler.complete_job(job['id'])
    scheduler.complete_job(job['id'])

    assert scheduler.available_chips(allocations[0]) == 64


def test_non_positive_chip_request_is_rejected(scheduler):
    with pytest.raises(ValueError, match="must be positive"):
        scheduler.submit_job('bad', 0)


# -- Placement policy -------------------------------------------------------


def test_preferred_chip_type_wins(scheduler):
    job = scheduler.submit_job('train', 8, preferred_chip='v4')

    assert job['assigned_chip'] == 'v4'


def test_falls_back_when_preferred_type_is_full(scheduler):
    scheduler.submit_job('fill', 64, preferred_chip='v6e')
    job = scheduler.submit_job('train', 8, preferred_chip='v6e')

    assert job['status'] == 'scheduled'
    assert job['assigned_chip'] == 'v4'


def test_tightest_fit_preserves_room_for_large_jobs(scheduler):
    """A small job should not eat the only zone big enough for a big one."""
    small = scheduler.submit_job('small', 8, preferred_chip='v4')
    big = scheduler.submit_job('big', 64, preferred_chip='v6e')

    assert small['assigned_zone'] == 'zone-b'
    assert big['status'] == 'scheduled'


# -- Reporting --------------------------------------------------------------


def test_capacity_report_covers_every_allocation(scheduler):
    assert len(scheduler.capacity_report()) == 2


def test_capacity_report_tracks_utilization(scheduler):
    scheduler.submit_job('train', 32, preferred_chip='v6e')
    row = next(r for r in scheduler.capacity_report() if r['zone'] == 'zone-a')

    assert row['chips_in_use'] == 32
    assert row['utilization'] == pytest.approx(0.5)


def test_total_compute_counts_job_states(scheduler):
    running = scheduler.submit_job('train', 64, preferred_chip='v6e')
    scheduler.submit_job('waiting', 64, preferred_chip='v6e')
    scheduler.complete_job(running['id'])

    report = scheduler.get_total_compute()
    assert report['jobs_completed'] == 1
    assert report['jobs_running'] == 1


def test_idle_fleet_reports_zero_utilization(scheduler):
    assert scheduler.get_total_compute()['utilization'] == 0.0


def test_completed_job_records_duration(scheduler):
    job = scheduler.submit_job('train', 8)
    scheduler.complete_job(job['id'])

    assert scheduler.completed_jobs[0]['duration'] >= 0


def test_total_tflops_sums_allocations(scheduler):
    # v6e is 918 PFLOPS/chip, v4 is 275
    assert scheduler.get_total_compute()['total_tflops'] == pytest.approx(
        64 * 918.0 + 32 * 275.0
    )
