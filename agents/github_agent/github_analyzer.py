from collections import Counter
from datetime import datetime


class GitHubAnalyzer:

    def analyze_languages(self, repos, language_maps):
        language_counter = Counter()

        for lang_map in language_maps:
            for lang, bytes_count in lang_map.items():
                language_counter[lang] += bytes_count

        return dict(language_counter)

    def analyze_activity(self, repos):
        last_updates = []

        for repo in repos:
            if repo.get("updated_at"):
                last_updates.append(
                    datetime.fromisoformat(
                        repo["updated_at"].replace("Z", "")
                    )
                )

        if not last_updates:
            return "inactive"

        days_since_last = (datetime.utcnow() - max(last_updates)).days

        if days_since_last < 30:
            return "highly_active"
        elif days_since_last < 90:
            return "moderately_active"
        else:
            return "low_activity"

    def estimate_skill_level(self, repos_count, stars, activity):
        if repos_count >= 35 and stars >= 2 and activity == "highly_active":
            return "advanced"
        if repos_count >= 16:
            return "intermediate"
        return "beginner"

    def analyze_projects(self, repos):
        difficulty = {"easy": 0, "medium": 0, "hard": 0}

        for repo in repos:
            size = repo.get("size", 0)
            stars = repo.get("stargazers_count", 0)

            if size < 500:
                difficulty["easy"] += 1
            elif size < 2000:
                difficulty["medium"] += 1
            else:
                difficulty["hard"] += 1

            if stars > 20:
                difficulty["hard"] += 1

        return difficulty
