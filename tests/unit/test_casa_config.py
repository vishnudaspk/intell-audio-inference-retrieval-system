"""
Unit tests for CASAConfig — V3.2 Speaker Intelligence
Validates weight constraints, threshold ordering, and config validation.
"""

import pytest
from services.casa_config import CASAConfig


class TestCASAConfigValidation:
    def test_default_config_is_valid(self):
        cfg = CASAConfig()  # should not raise
        assert cfg.enable_casa is True

    def test_normal_weights_sum_to_one(self):
        cfg = CASAConfig()
        total = cfg.w_acoustic + cfg.w_temporal + cfg.w_linguistic
        assert abs(total - 1.0) < 1e-6

    def test_short_utt_weights_sum_to_one(self):
        cfg = CASAConfig()
        total = cfg.short_utt_w_acoustic + cfg.short_utt_w_temporal + cfg.short_utt_w_linguistic
        assert abs(total - 1.0) < 1e-6

    def test_confirm_threshold_above_uncertain_threshold(self):
        cfg = CASAConfig()
        assert cfg.uncertain_threshold < cfg.confirm_threshold

    def test_thresholds_in_range(self):
        cfg = CASAConfig()
        assert 0.0 <= cfg.uncertain_threshold <= 1.0
        assert 0.0 <= cfg.confirm_threshold <= 1.0

    def test_invalid_normal_weights_raise(self):
        with pytest.raises(ValueError, match="Normal fusion weights"):
            CASAConfig(w_acoustic=0.60, w_temporal=0.25, w_linguistic=0.20)  # sums to 1.05

    def test_invalid_short_utt_weights_raise(self):
        with pytest.raises(ValueError, match="Short-utterance fusion weights"):
            CASAConfig(short_utt_w_acoustic=0.50, short_utt_w_temporal=0.40, short_utt_w_linguistic=0.20)

    def test_invalid_threshold_ordering_raises(self):
        with pytest.raises(ValueError, match="Thresholds"):
            CASAConfig(uncertain_threshold=0.80, confirm_threshold=0.70)

    def test_custom_weights_accepted(self):
        cfg = CASAConfig(w_acoustic=0.50, w_temporal=0.30, w_linguistic=0.20)
        assert abs(cfg.w_acoustic + cfg.w_temporal + cfg.w_linguistic - 1.0) < 1e-6

    def test_filler_words_populated(self):
        cfg = CASAConfig()
        assert "yeah" in cfg.filler_words
        assert "okay" in cfg.filler_words
        assert "oh" in cfg.filler_words

    def test_question_markers_populated(self):
        cfg = CASAConfig()
        assert "what" in cfg.question_markers
        assert "who" in cfg.question_markers
        assert "how" in cfg.question_markers

    def test_early_dialogue_window_positive(self):
        cfg = CASAConfig()
        assert cfg.early_dialogue_window_sec > 0

    def test_short_utt_max_words_positive(self):
        cfg = CASAConfig()
        assert cfg.short_utt_max_words > 0

    def test_casa_disabled_config(self):
        cfg = CASAConfig(enable_casa=False)
        assert cfg.enable_casa is False
