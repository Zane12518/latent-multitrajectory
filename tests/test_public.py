import importlib.util
from pathlib import Path


def test_public_evidence_and_links():
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('check_public',root/'scripts/check_public.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.check()['gru_pairs']==12
