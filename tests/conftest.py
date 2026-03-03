# tests/conftest.py
"""
Shared pytest fixtures for miRPV-NG tests.
"""

import pytest


# ==================== Sample Sequences ====================

@pytest.fixture
def simple_hairpin_seq():
    """A simple 70nt hairpin-like sequence."""
    return "CCUGCCGACUAUGCCAAUUGUCAGGUCCCAACCUGGGCUGGGGUCCGAGGGGAGUUGGUAGAUGGGUGG"


@pytest.fixture
def simple_hairpin_struct():
    """Expected structure for the simple hairpin (pre-computed)."""
    # This is a realistic hairpin structure with paired brackets
    return "(((((((.((((((((((((((......))))))))...........))))))))))))......."


@pytest.fixture
def simple_hairpin_mfe():
    """Example MFE value."""
    return -25.5


@pytest.fixture
def short_seq():
    """A short sequence (too short for hairpin)."""
    return "ACGUACGU"


@pytest.fixture
def short_struct():
    """Structure for short sequence."""
    return "........"


@pytest.fixture
def valid_mirna_seq():
    """Known pre-miRNA-like sequence from examples."""
    return "ATCCTGCCGACTATGCCAATTGTCAGGTCCCAACCTGGGCTGGGGTCCGAGGGGAGTTGGTAGATGGGTGGCAGGTAGT"


# ==================== Sample FASTA Content ====================

@pytest.fixture
def sample_fasta_content():
    """Sample FASTA file content for testing read_fasta."""
    return """>seq1
ACGUACGUACGU
>seq2
GCAUGCAUGCAU
>seq3 with description
UUUAAACCCGGG
"""


# ==================== Model Path ====================

@pytest.fixture
def rf_model_path():
    """Path to the pre-trained RF model."""
    from pathlib import Path
    return Path(__file__).parent.parent / "models" / "hsa_premirna_rf_extended_tier2_v6.pkl"


@pytest.fixture
def mature_model_path():
    """Path to the mature XGBoost ranker model."""
    from pathlib import Path
    return Path(__file__).parent.parent / "models" / "hsa_mature_xgbrank_len18_26_v4.pkl"
