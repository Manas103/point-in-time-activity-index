import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from activityindex import vintage_store


@pytest.fixture
def memory_con():
    con = vintage_store.connect(":memory:")
    yield con
    con.close()
