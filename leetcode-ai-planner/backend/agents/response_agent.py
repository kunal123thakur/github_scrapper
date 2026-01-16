def build_response(schedule_map):
    return {
        "total_days": len(schedule_map),
        "schedule": [
            {"date": d, "questions": qs}
            for d, qs in sorted(schedule_map.items())
        ]
    }
