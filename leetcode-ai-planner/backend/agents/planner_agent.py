def plan_distribution(intent: dict) -> dict:
    topics = intent["topics"]

    explicit_total = sum(
        t["count"] for t in topics.values() if t["count"] is not None
    )

    remaining_topics = [
        k for k, v in topics.items() if v["count"] is None
    ]

    if intent.get("total_questions"):
        remaining = intent["total_questions"] - explicit_total

        if remaining < 0:
            raise ValueError("Topic-wise counts exceed total questions")

        if remaining_topics:
            per_topic = remaining // len(remaining_topics)
            for t in remaining_topics:
                topics[t]["count"] = per_topic

    return {
        "topics": topics,
        "duration_days": intent.get("duration_days", 21)
    }
