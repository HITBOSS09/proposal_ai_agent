# Defence Proposal AI Agent

A clean, production-ready Python project scaffold for a Defence Proposal AI Agent.

## Overview

This repository is designed for a functional AI architecture that separates concerns by feature area rather than by technical layers.

## Project structure

- `src/proposal_ai_agent/` - main package for application code.
  - `ingestion/` - document ingestion, loading, parsing, and chunking workflows.
  - `retrieval/` - search and retrieval mechanisms, vector search, and ranking.
  - `generation/` - prompt construction and generation workflows.
  - `export/` - proposal export formats, document generation, and output handling.
  - `prompts/` - prompt templates and system prompt management.
  - `utils/` - shared utilities, logging, and helper functions.
- `tests/` - test package for unit and integration tests.
- `data/raw/` - source knowledge base documents.
- `logs/` - runtime logs and audit artifacts.
- `config.py` - root-level configuration placeholders.
- `requirements.txt` - initial dependencies for development.
- `.gitignore` - ignores Python artifacts, environment files, and editor metadata.

## Current scope

Phase 0 includes a single source document:

- `data/raw/Proposal Draft.docx`

## Python version

Target runtime: Python 3.12

## Next steps

- Add module implementations in each functional package
- Add configuration models in `config.py`
- Add tests under `tests/`
