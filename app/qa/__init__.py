# -*- coding: utf-8 -*-
"""In-image QA suite: unit tests, smoke/warmup checks and load tests.

Usage (inside the image or host venv):
    python -m app.qa.run_qa --unit            # fast unit tests (no GPU/model)
    python -m app.qa.run_qa --smoke           # functional checks against a running service
    python -m app.qa.run_qa --load            # concurrency load test
    python -m app.qa.run_qa --all             # unit + smoke
"""
