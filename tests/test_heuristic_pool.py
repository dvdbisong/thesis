"""
Unit tests for HeuristicPool and DLModelHeuristic.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.heuristics.base import HeuristicBase
from src.heuristics.dl.heuristic_pool import DLModelHeuristic, HeuristicPool


class TestHeuristicPoolDiscovery:
    """Tests for model discovery functionality."""

    def test_pool_initialization(self):
        """Test that pool initializes without loading models."""
        pool = HeuristicPool(load_on_init=False)
        assert isinstance(pool, HeuristicPool)
        assert len(pool._heuristics) == 0

    def test_pool_discovers_models(self):
        """Test that pool discovers .pth files in models directory."""
        pool = HeuristicPool(load_on_init=False)
        models = pool.list_models()
        # Should find at least some models
        assert isinstance(models, list)
        # Model IDs should start with 'dl_'
        for m in models:
            assert m.startswith("dl_")

    def test_pool_repr(self):
        """Test string representation."""
        pool = HeuristicPool(load_on_init=False)
        repr_str = repr(pool)
        assert "HeuristicPool" in repr_str
        assert "models loaded" in repr_str


class TestHeuristicPoolMetadata:
    """Tests for metadata functionality."""

    @pytest.fixture
    def pool(self):
        """Create pool without loading models."""
        return HeuristicPool(load_on_init=False)

    def test_get_metadata(self, pool):
        """Test metadata retrieval for discovered models."""
        models = pool.list_models()
        if models:
            info = pool.get_metadata(models[0])
            assert "site" in info
            assert "path" in info
            assert "encoder" in info

    def test_get_metadata_unknown_model(self, pool):
        """Test that unknown model raises KeyError."""
        with pytest.raises(KeyError):
            pool.get_metadata("nonexistent_model")


class TestModelLoading:
    """Tests for model loading functionality."""

    @pytest.fixture
    def pool_with_models(self):
        """Create pool and load all models."""
        pool = HeuristicPool(load_on_init=True)
        return pool

    def test_load_all_models(self, pool_with_models):
        """Test loading all discovered models."""
        heuristics = pool_with_models.get_all_heuristics()
        assert len(heuristics) > 0
        for h in heuristics:
            assert isinstance(h, HeuristicBase)

    def test_get_heuristic(self, pool_with_models):
        """Test getting specific heuristic by ID."""
        models = pool_with_models.list_models()
        if models:
            h = pool_with_models.get_heuristic(models[0])
            assert isinstance(h, DLModelHeuristic)
            assert h.name == models[0]

    def test_get_site_model(self, pool_with_models):
        """Test getting model by site code."""
        # Try to get a model for a known site
        for site in ["UCA", "UCU"]:
            h = pool_with_models.get_site_model(site)
            if h is not None:
                assert isinstance(h, DLModelHeuristic)
                assert site in h.name
                break


class TestInference:
    """Tests for model inference functionality."""

    @pytest.fixture
    def pool(self):
        """Create pool with models loaded."""
        return HeuristicPool(load_on_init=True)

    @pytest.fixture
    def dummy_tile_10ch(self):
        """Create dummy 10-channel tile."""
        return torch.randn(10, 512, 512)

    @pytest.fixture
    def dummy_tile_12ch(self):
        """Create dummy 12-channel tile (with aux bands)."""
        return torch.randn(12, 512, 512)

    def test_predict_10_channels(self, pool, dummy_tile_10ch):
        """Test prediction with 10-channel input."""
        heuristics = pool.get_all_heuristics()
        if heuristics:
            h = heuristics[0]
            mask = h.predict(dummy_tile_10ch)
            assert mask.shape == (512, 512)
            assert mask.dtype == torch.float32
            # Should be binary
            unique = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_predict_12_channels(self, pool, dummy_tile_12ch):
        """Test prediction with 12-channel input (aux bands stripped)."""
        heuristics = pool.get_all_heuristics()
        if heuristics:
            h = heuristics[0]
            mask = h.predict(dummy_tile_12ch)
            assert mask.shape == (512, 512)

    def test_predict_proba(self, pool, dummy_tile_10ch):
        """Test probability prediction."""
        heuristics = pool.get_all_heuristics()
        if heuristics:
            h = heuristics[0]
            proba = h.predict_proba(dummy_tile_10ch)
            assert proba.shape == (512, 512)
            assert proba.min() >= 0.0
            assert proba.max() <= 1.0

    def test_predict_invalid_channels(self, pool):
        """Test that invalid channel count raises error."""
        heuristics = pool.get_all_heuristics()
        if heuristics:
            h = heuristics[0]
            bad_tile = torch.randn(5, 512, 512)
            with pytest.raises(ValueError):
                h.predict(bad_tile)

    def test_predict_with_all(self, pool, dummy_tile_10ch):
        """Test prediction with all models."""
        results = pool.predict_with_all(dummy_tile_10ch)
        assert isinstance(results, dict)
        for model_id, mask in results.items():
            assert mask.shape == (512, 512)


class TestEnsemble:
    """Tests for ensemble prediction functionality."""

    @pytest.fixture
    def pool(self):
        """Create pool with models loaded."""
        return HeuristicPool(load_on_init=True)

    @pytest.fixture
    def dummy_tile(self):
        """Create dummy tile."""
        return torch.randn(10, 512, 512)

    def test_ensemble_mean(self, pool, dummy_tile):
        """Test mean ensemble method."""
        mask = pool.predict_ensemble(dummy_tile, method="mean")
        assert mask.shape == (512, 512)

    def test_ensemble_vote(self, pool, dummy_tile):
        """Test voting ensemble method."""
        mask = pool.predict_ensemble(dummy_tile, method="vote")
        assert mask.shape == (512, 512)

    def test_ensemble_weighted(self, pool, dummy_tile):
        """Test weighted ensemble method."""
        models = pool.list_models()
        weights = {m: 1.0 for m in models}
        mask = pool.predict_ensemble(dummy_tile, method="weighted", weights=weights)
        assert mask.shape == (512, 512)

    def test_ensemble_invalid_method(self, pool, dummy_tile):
        """Test that invalid ensemble method raises error."""
        with pytest.raises(ValueError):
            pool.predict_ensemble(dummy_tile, method="invalid")


class TestHeuristicInterface:
    """Tests for HeuristicBase interface compliance."""

    @pytest.fixture
    def heuristic(self):
        """Get a loaded heuristic."""
        pool = HeuristicPool(load_on_init=True)
        heuristics = pool.get_all_heuristics()
        return heuristics[0] if heuristics else None

    def test_has_name(self, heuristic):
        """Test that heuristic has name attribute."""
        if heuristic:
            assert hasattr(heuristic, "name")
            assert isinstance(heuristic.name, str)

    def test_has_predict(self, heuristic):
        """Test that heuristic has predict method."""
        if heuristic:
            assert hasattr(heuristic, "predict")
            assert callable(heuristic.predict)

    def test_get_config(self, heuristic):
        """Test get_config method."""
        if heuristic:
            config = heuristic.get_config()
            assert isinstance(config, dict)
            assert "name" in config
            assert "type" in config

    def test_requires_training(self, heuristic):
        """Test requires_training method."""
        if heuristic:
            # DL models don't require training (already trained)
            assert not heuristic.requires_training()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
