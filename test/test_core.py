# test/test_core.py

def test_search_space_shape():
    """Minimal smoke test to ensure SEARCH_SPACE is defined and has expected keys."""
    from llama_optimus.core import SEARCH_SPACE
    assert isinstance(SEARCH_SPACE, dict)
    assert 'batch_size' in SEARCH_SPACE

