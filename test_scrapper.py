# from agents.github_agent.github_scraper import GitHubScraper

# scraper = GitHubScraper()

# profile = scraper.get_user_profile("kunal123thakur")
# repos = scraper.get_user_repos("kunal123thakur")

# print("Username:", profile["login"])
# print("Repo count:", len(repos))
# print("First repo:", repos[0]["name"])


from agents.github_agent.github_agent import GitHubProfileAgent
import json

agent = GitHubProfileAgent()

result = agent.run("kunal123thakur")

print(json.dumps(result, indent=2))
