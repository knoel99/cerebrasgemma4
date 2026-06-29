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