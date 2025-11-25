from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str = "google/bert_uncased_L-4_H-256_A-4"):
    """
    Create token classification model optimized for speed and accuracy.
    
    Recommended models for latency vs accuracy trade-off:
    - "distilbert-base-uncased": Best accuracy, ~35-45ms (too slow)
    - "google/bert_uncased_L-4_H-256_A-4": Good balance, ~10-15ms
    - "google/bert_uncased_L-2_H-128_A-2": Fastest, ~5-8ms
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Token classification model
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # Reduced dropout for small models to retain learning capacity
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # Handle vocab size differences
    )
    
    return model