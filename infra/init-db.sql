-- Enable pgvector extension and configure read-only Postgres role for Ledger Analyst
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'finvigil_readonly') THEN
        CREATE ROLE finvigil_readonly WITH LOGIN PASSWORD 'finvigil_ro_secret';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE finvigil TO finvigil_readonly;
GRANT USAGE ON SCHEMA public TO finvigil_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO finvigil_readonly;
