"""Unit tests for modularml.core.data.schema_constants module."""

import pytest

from modularml.core.data.schema_constants import (
    ALL_DOMAINS,
    ALL_REPS,
    DOMAIN_FEATURES,
    DOMAIN_OUTPUTS,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    INVALID_LABEL_CHARACTERS,
    REP_RAW,
    REP_TRANSFORMED,
    ROLE_ANCHOR,
    ROLE_DEFAULT,
    ROLE_NEGATIVE,
    ROLE_PAIR,
    ROLE_POSITIVE,
    STREAM_DEFAULT,
)


# ---------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_domain_values():
    """Test that domain constants have expected string values."""
    assert DOMAIN_FEATURES == "features"
    assert DOMAIN_TARGETS == "targets"
    assert DOMAIN_TAGS == "tags"
    assert DOMAIN_SAMPLE_UUIDS == "sample_uuids"
    assert DOMAIN_OUTPUTS == "outputs"


@pytest.mark.unit
def test_all_domains_contains_all():
    """Test that ALL_DOMAINS tuple contains all domain constants."""
    assert DOMAIN_FEATURES in ALL_DOMAINS
    assert DOMAIN_TARGETS in ALL_DOMAINS
    assert DOMAIN_TAGS in ALL_DOMAINS
    assert DOMAIN_SAMPLE_UUIDS in ALL_DOMAINS
    assert DOMAIN_OUTPUTS in ALL_DOMAINS
    assert len(ALL_DOMAINS) == 5


@pytest.mark.unit
def test_all_domains_is_tuple():
    """Test that ALL_DOMAINS is a tuple."""
    assert isinstance(ALL_DOMAINS, tuple)


# ---------------------------------------------------------------------
# Representation constants
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_rep_values():
    """Test that representation constants have expected string values."""
    assert REP_RAW == "raw"
    assert REP_TRANSFORMED == "transformed"


@pytest.mark.unit
def test_all_reps_contains_all():
    """Test that ALL_REPS tuple contains all rep constants."""
    assert REP_RAW in ALL_REPS
    assert REP_TRANSFORMED in ALL_REPS
    assert len(ALL_REPS) == 2


@pytest.mark.unit
def test_all_reps_is_tuple():
    """Test that ALL_REPS is a tuple."""
    assert isinstance(ALL_REPS, tuple)


# ---------------------------------------------------------------------
# Sampler vocabulary
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_stream_default():
    """Test stream default constant."""
    assert STREAM_DEFAULT == "default"


@pytest.mark.unit
def test_role_constants():
    """Test that role constants have expected string values."""
    assert ROLE_DEFAULT == "default"
    assert ROLE_ANCHOR == "anchor"
    assert ROLE_PAIR == "pair"
    assert ROLE_POSITIVE == "positive"
    assert ROLE_NEGATIVE == "negative"
