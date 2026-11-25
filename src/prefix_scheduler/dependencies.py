"""Compute immediate unlocks from a materialized snapshot of a request DAG."""

def immediate_unlocks(candidate, requests):
    """Count blocked children whose other parents have successfully completed.

    Missing parents, failed parents, and unmaterialized future children do not
    count. The caller must refresh the snapshot after state transitions.
    """
    by_id = {r.request_id: r for r in requests}
    if len(by_id) != len(requests): raise ValueError('duplicate request ID')
    total = 0
    for child in requests:
        if child.status != 'blocked' or candidate.request_id not in child.parent_ids:
            continue
        others = [parent for parent in child.parent_ids if parent != candidate.request_id]
        if all(parent in by_id and by_id[parent].status == 'completed' for parent in others):
            total += 1
    return total
