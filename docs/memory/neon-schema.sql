-- Version 1. Additive bootstrap in the existing dedicated GB10X memory database.
-- No application/runtime database or legacy memory is modified.
CREATE TABLE IF NOT EXISTS public.gb10x_memory_state (
    project_key text PRIMARY KEY CHECK (project_key = 'gb10x'),
    revision bigint NOT NULL CHECK (revision > 0),
    bundle_sha256 text NOT NULL CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    checkpoint jsonb NOT NULL CHECK (jsonb_typeof(checkpoint) = 'object'),
    updated_at timestamptz NOT NULL DEFAULT now()
);
