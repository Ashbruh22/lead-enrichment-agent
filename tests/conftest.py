from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def company_html() -> str:
    return (FIXTURES / "company_page.html").read_text(encoding="utf-8")
