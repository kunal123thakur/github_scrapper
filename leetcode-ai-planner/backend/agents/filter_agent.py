import re

def filter_questions(df, plan):
    result = []

    for topic, cfg in plan["topics"].items():
        if topic.lower() in ["any", "dsa", "all", "questions", "leetcode"]:
            temp = df
        else:
            temp = df[df["tags"].apply(lambda t: topic.lower() in t)]

        if cfg["difficulty"] != "any":
            temp = temp[temp["difficulty"] == cfg["difficulty"]]

        if len(temp) < cfg["count"]:
            raise ValueError(
                f"Only {len(temp)} questions available for {topic}, "
                f"but {cfg['count']} requested."
            )

        result.append(temp.sample(cfg["count"]))

    return result
