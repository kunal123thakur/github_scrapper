from datetime import date, timedelta

def schedule(dfs, days):
    questions = []
    for df in dfs:
        questions.extend(df.to_dict(orient="records"))

    start = date.today()
    schedule_map = {}

    for i, q in enumerate(questions):
        day = start + timedelta(days=i % days)
        schedule_map.setdefault(str(day), []).append({
            "title": q["task_id"].replace("-", " ").title(),
            "difficulty": q["difficulty"].title(),
            "topic": q["tags"][0].title(),
            "url": f"https://leetcode.com/problems/{q['task_id']}/"
        })

    return schedule_map
