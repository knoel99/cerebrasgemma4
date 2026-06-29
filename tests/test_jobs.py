from cerebrasgemma4.api.routes.jobs import _estimate_remaining_sec, _job_to_status
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore


def test_list_jobs_newest_first(tmp_path):
    store = JobStore(root=tmp_path)
    first = store.create()
    store.update(first.job_id, status=JobStatus.COMPLETED, title="First")
    second = store.create()
    store.update(second.job_id, status=JobStatus.COMPLETED, title="Second")

    jobs = store.list_jobs()
    assert len(jobs) == 2
    assert jobs[0].job_id == second.job_id
    assert jobs[1].job_id == first.job_id


def test_delete_job(tmp_path):
    store = JobStore(root=tmp_path)
    record = store.create()
    store.update(
        record.job_id,
        status=JobStatus.COMPLETED,
        source_name="demo.mp4",
        source_type="file",
        title="Demo report",
    )

    store.delete(record.job_id)

    try:
        store.load(record.job_id)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_job_record_backward_compatible(tmp_path):
    store = JobStore(root=tmp_path)
    record = store.create()
    meta = store.meta_path(record.job_id)
    meta.write_text(
        '{"job_id": "%s", "status": "completed", "progress": 100, "message": "Done"}'
        % record.job_id,
        encoding="utf-8",
    )

    loaded = store.load(record.job_id)
    assert loaded.status == JobStatus.COMPLETED
    assert loaded.title is None
    assert loaded.source_type is None


def test_job_status_includes_language(tmp_path):
    from cerebrasgemma4.api.routes.jobs import _job_to_status

    store = JobStore(root=tmp_path)
    record = store.create()
    store.update(record.job_id, status=JobStatus.COMPLETED, language="fr")

    status = _job_to_status(store.load(record.job_id))
    assert status.language == "fr"


def test_job_status_includes_custom_prompt(tmp_path):
    from cerebrasgemma4.api.routes.jobs import _job_to_status

    store = JobStore(root=tmp_path)
    record = store.create()
    store.update(
        record.job_id,
        status=JobStatus.COMPLETED,
        custom_prompt="Summarize for executives only.",
    )

    status = _job_to_status(store.load(record.job_id))
    assert status.custom_prompt == "Summarize for executives only."


def test_job_status_includes_eta(tmp_path):
    store = JobStore(root=tmp_path)
    record = store.create()
    store.update(
        record.job_id,
        status=JobStatus.ANALYZING,
        progress=70,
        message="Analyzing frames",
        metrics={"estimated_total_minutes": 5.0},
    )
    loaded = store.load(record.job_id)

    remaining = _estimate_remaining_sec(loaded)
    assert remaining is not None
    assert 0.0 <= remaining <= 5.0 * 60.0

    status = _job_to_status(loaded)
    assert status.estimated_total_minutes == 5.0
    assert status.estimated_remaining_sec is not None