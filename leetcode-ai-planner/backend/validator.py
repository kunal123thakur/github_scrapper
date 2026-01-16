def validate_intent(intent):
    if not intent.get("topics"):
        raise ValueError("No valid topics found in request")

    for topic, cfg in intent["topics"].items():
        if cfg["count"] is not None and cfg["count"] <= 0:
            raise ValueError(f"Invalid count for {topic}")
