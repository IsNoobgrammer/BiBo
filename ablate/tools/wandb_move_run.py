"""Move W&B runs between projects in the same entity.

The public Python API cannot do this: `Api().runs(...)[i].project` is a plain str, and `Run.update()`
only persists tags/description/notes/config/group/jobType -- there is no project field in its
mutation. The BACKEND does expose `moveRuns`, so this drives that directly through the same
authenticated GraphQL client the public API already holds.

    python -m ablate.tools.wandb_move_run --entity ablations-tinycompany-ai \
        --src bibo-attnres-v2 --dst bibo-attnres-1b --runs 8zl14c8v,abc12345

Refuses to run unless the destination project already exists, because moveRuns against a missing
project reports success and the runs land nowhere visible. W&B creates a project on the first run
that logs to it, so start an arm there first (or pass --allow_missing_dst if you know better).
"""
import argparse
import json
import sys

import wandb

MUTATION = """
mutation MoveRuns($input: MoveRunsInput!) {
  moveRuns(input: $input) { clientMutationId }
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--src", required=True, help="source project name")
    ap.add_argument("--dst", required=True, help="destination project name")
    ap.add_argument("--runs", required=True, help="comma-separated run IDs (the short hash, not the display name)")
    ap.add_argument("--allow_missing_dst", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    api = wandb.Api()

    # Resolve the runs first: a typo'd ID would otherwise move nothing and report success.
    src_runs = {r.id: r for r in api.runs(f"{args.entity}/{args.src}")}
    missing = [i for i in ids if i not in src_runs]
    if missing:
        raise SystemExit(f"not in {args.entity}/{args.src}: {missing}\n"
                         f"available: {sorted(src_runs)}")
    for i in ids:
        print(f"  will move {i}  {src_runs[i].name[:70]}  (state={src_runs[i].state})")

    if not args.allow_missing_dst:
        try:
            api.runs(f"{args.entity}/{args.dst}")[:1]
        except Exception as e:
            raise SystemExit(
                f"destination {args.entity}/{args.dst} not reachable ({type(e).__name__}). "
                f"W&B creates a project on first write -- launch an arm there first, or pass "
                f"--allow_missing_dst.")

    if args.dry_run:
        print("dry run, nothing moved")
        return

    payload = {
        "sourceEntityName": args.entity,
        "sourceProjectName": args.src,
        "destinationEntityName": args.entity,
        "destinationProjectName": args.dst,
        "filters": json.dumps({"name": {"$in": ids}}),
    }
    # _exec is the authenticated GraphQL client behind the public Api; any run object carries it.
    src_runs[ids[0]]._exec(MUTATION, input=payload)

    # VERIFY. The mutation returns only a clientMutationId, so a silent no-op is indistinguishable
    # from success without re-reading both sides.
    # A FRESH Api: wandb.Api() memoizes run listings per project, so reusing `api` here reports the
    # pre-move state and turns a successful move into a false FAILED. Cost me one confused report.
    fresh = wandb.Api()
    still = {r.id for r in fresh.runs(f"{args.entity}/{args.src}")} & set(ids)
    landed = {r.id for r in fresh.runs(f"{args.entity}/{args.dst}")} & set(ids)
    print(f"\nmoved  : {sorted(landed)}")
    print(f"stuck  : {sorted(still)}")
    if still or landed != set(ids):
        sys.exit(f"FAILED: expected {sorted(ids)} in {args.dst}")
    print("OK")


if __name__ == "__main__":
    main()
