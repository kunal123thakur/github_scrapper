from agents.github_agent.github_scraper import GitHubScraper
from agents.github_agent.github_analyzer import GitHubAnalyzer


class GitHubProfileAgent:
    """
    Profile Intelligence Sub-Agent: GitHub
    """

    def __init__(self):
        self.scraper = GitHubScraper()
        self.analyzer = GitHubAnalyzer()

    def run(self, github_username: str) -> dict:
        profile = self.scraper.get_user_profile(github_username)
        repos = self.scraper.get_user_repos(github_username)

        language_maps = []
        for repo in repos:
            lang_map = self.scraper.get_repo_languages(
                github_username, repo["name"]
            )
            language_maps.append(lang_map)

        languages = self.analyzer.analyze_languages(repos, language_maps)
        activity = self.analyzer.analyze_activity(repos)
        project_difficulty = self.analyzer.analyze_projects(repos)

        skill_level = self.analyzer.estimate_skill_level(
            repos_count=len(repos),
            stars=profile.get("followers", 0),
            activity=activity
        )

        return {
            "github_username": github_username,
            "profile_summary": {
                "bio": profile.get("bio"),
                "public_repos": profile.get("public_repos"),
                "followers": profile.get("followers"),
                "following": profile.get("following")
            },
            "languages_used": languages,
            "activity_level": activity,
            "project_difficulty_distribution": project_difficulty,
            "inferred_skill_level": skill_level
        }
