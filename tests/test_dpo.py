"""Tests for DPO (Direct Preference Optimization) implementation."""

import pytest
import torch
from modal_runtime.utils.preference_utils import format_preference_pairs_for_dpo
from api.schemas import TrainingExample
from pydantic import ValidationError


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(
        self, texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
    ):
        """Mock tokenization that returns simple integer sequences."""
        # Simple mock: convert text length to token IDs
        token_ids = []
        attention_masks = []

        max_len = max(len(text) for text in texts)
        for text in texts:
            # Create token IDs based on text length (mock)
            ids = list(range(len(text)))
            # Pad to max length
            if len(ids) < max_len:
                ids.extend([0] * (max_len - len(ids)))
            token_ids.append(ids)

            # Create attention mask
            mask = [1] * len(text) + [0] * (max_len - len(text))
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.tensor(attention_masks),
        }


class TestPreferenceUtils:
    """Test preference_utils.py functions."""

    def test_format_preference_pairs_basic(self):
        """Test basic preference pair formatting."""
        preference_pairs = [
            {
                "prompt": "What is Python?",
                "chosen": "Python is a high-level programming language.",
                "rejected": "Python is a snake.",
            }
        ]

        tokenizer = MockTokenizer()
        result = format_preference_pairs_for_dpo(
            preference_pairs=preference_pairs,
            tokenizer=tokenizer,
            max_seq_length=128,
        )

        # Check that result has expected keys
        assert "chosen_input_ids" in result
        assert "chosen_attention_mask" in result
        assert "rejected_input_ids" in result
        assert "rejected_attention_mask" in result

        # Check shapes
        assert result["chosen_input_ids"].shape[0] == 1  # batch size
        assert result["rejected_input_ids"].shape[0] == 1

        # Check that tensors are returned
        assert isinstance(result["chosen_input_ids"], torch.Tensor)
        assert isinstance(result["rejected_input_ids"], torch.Tensor)

    def test_format_multiple_pairs(self):
        """Test formatting multiple preference pairs."""
        preference_pairs = [
            {
                "prompt": "Question 1?",
                "chosen": "Good answer 1",
                "rejected": "Bad answer 1",
            },
            {
                "prompt": "Question 2?",
                "chosen": "Good answer 2",
                "rejected": "Bad answer 2",
            },
        ]

        tokenizer = MockTokenizer()
        result = format_preference_pairs_for_dpo(
            preference_pairs=preference_pairs,
            tokenizer=tokenizer,
            max_seq_length=128,
        )

        # Check batch size
        assert result["chosen_input_ids"].shape[0] == 2
        assert result["rejected_input_ids"].shape[0] == 2


class TestDPOSchemas:
    """Test schema validation for DPO format."""

    def test_valid_dpo_format(self):
        """Test valid DPO preference pair."""
        example = TrainingExample(
            prompt="What is the capital of France?",
            chosen="Paris is the capital of France.",
            rejected="I don't know.",
        )

        assert example.prompt == "What is the capital of France?"
        assert example.chosen == "Paris is the capital of France."
        assert example.rejected == "I don't know."

    def test_valid_sft_text_format(self):
        """Test valid SFT text format still works."""
        example = TrainingExample(
            text="This is a training example.",
        )

        assert example.text == "This is a training example."
        assert example.prompt is None
        assert example.chosen is None
        assert example.rejected is None

    def test_valid_sft_messages_format(self):
        """Test valid SFT messages format still works."""
        example = TrainingExample(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )

        assert len(example.messages) == 2
        assert example.text is None
        assert example.prompt is None

    def test_missing_all_fields_raises_error(self):
        """Test that missing all fields raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample()

        assert "Must provide one of" in str(exc_info.value)

    def test_partial_dpo_fields_raises_error(self):
        """Test that partial DPO fields raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample(
                prompt="What is Python?",
                chosen="A programming language",
                # Missing 'rejected'
            )

        # The validation error should indicate that a valid format is required
        error_str = str(exc_info.value)
        assert "Must provide one of" in error_str or "ALL three fields" in error_str

    def test_multiple_formats_raises_error(self):
        """Test that providing multiple formats raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample(
                text="Some text",
                prompt="A prompt",
                chosen="Chosen",
                rejected="Rejected",
            )

        assert "only ONE format" in str(exc_info.value)

    def test_empty_dpo_fields_raises_error(self):
        """Test that empty DPO fields raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample(
                prompt="   ",  # Whitespace only
                chosen="Valid response",
                rejected="Another response",
            )

        assert "cannot be empty" in str(exc_info.value)


class TestDPOLossFunction:
    """Test DPO loss function."""

    def test_dpo_loss_import(self):
        """Test that DPO loss function can be imported."""
        from modal_runtime.loss_functions import get_loss_function, LOSS_FUNCTIONS

        assert "dpo" in LOSS_FUNCTIONS
        loss_fn = get_loss_function("dpo")
        assert loss_fn is not None

    def test_dpo_loss_signature(self):
        """Test DPO loss function signature."""
        from modal_runtime.loss_functions import dpo_loss
        import inspect

        sig = inspect.signature(dpo_loss)
        params = sig.parameters

        assert "model" in params
        assert "batch" in params
        assert "beta" in params
        assert "reference_model" in params


class TestDPOIntegration:
    """Integration tests for complete DPO workflow."""

    def test_tokenization_to_loss_flow(self):
        """Test full flow from preference pairs to tokenized batch."""
        from modal_runtime.utils.tokenization import tokenize_batch

        # Create preference pair data
        batch_data = [
            {
                "prompt": "What is 2+2?",
                "chosen": "2+2 equals 4.",
                "rejected": "I don't know.",
            }
        ]

        tokenizer = MockTokenizer()

        # Tokenize with DPO loss function
        result = tokenize_batch(
            batch_data=batch_data,
            tokenizer=tokenizer,
            max_seq_length=128,
            loss_fn="dpo",
        )

        # Check that DPO-specific keys are present
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result

        # Check that tokenization worked
        assert result["chosen_input_ids"].shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
