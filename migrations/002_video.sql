-- D2 video moderation. Idempotent: safe to re-run. Only adds; never removes or changes 001_init.sql.
-- Applied by the `migrate` service in the compose files, after 001_init.sql.

-- Which waiting list a job belongs to. 'fast' is every D1 job; 'video' jobs run on the video loop.
ALTER TABLE job ADD COLUMN IF NOT EXISTS lane text NOT NULL DEFAULT 'fast';
ALTER TABLE job DROP CONSTRAINT IF EXISTS job_lane_check;
ALTER TABLE job ADD CONSTRAINT job_lane_check CHECK (lane IN ('fast', 'video'));

CREATE INDEX IF NOT EXISTS job_lane_claim_idx ON job (lane, status, next_attempt_at);

-- One row per kept frame of a video, with its scores. Kept for the appeal and review workflow.
-- time_ms is the position in the video, so a reviewer can jump to the exact moment.
CREATE TABLE IF NOT EXISTS video_frame (
    job_id         uuid        NOT NULL REFERENCES job (id) ON DELETE CASCADE,
    media_id       text        NOT NULL,
    frame_no       integer     NOT NULL CHECK (frame_no >= 0),
    time_ms        integer     NOT NULL CHECK (time_ms >= 0),
    source         text        NOT NULL CHECK (source IN ('sample', 'scene')),
    scores         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    flagged        boolean     NOT NULL DEFAULT false,
    evidence_path  text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, media_id, frame_no)
);

CREATE INDEX IF NOT EXISTS video_frame_flagged_idx ON video_frame (flagged, created_at) WHERE flagged;
