import requests
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"


class GitHubScraper:
    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {GITHUB_TOKEN}"
        }

    def get_user_profile(self, username: str) -> dict:
        url = f"{BASE_URL}/users/{username}"
        res = requests.get(url, headers=self.headers)

        if res.status_code != 200:
            raise Exception("Failed to fetch GitHub profile")

        return res.json()

    def get_user_repos(self, username: str) -> list:
        url = f"{BASE_URL}/users/{username}/repos?per_page=100"
        res = requests.get(url, headers=self.headers)

        if res.status_code != 200:
            raise Exception("Failed to fetch repositories")

        return res.json()

    def get_repo_languages(self, owner: str, repo: str) -> dict:
        url = f"{BASE_URL}/repos/{owner}/{repo}/languages"
        res = requests.get(url, headers=self.headers)

        if res.status_code != 200:
            return {}

        return res.json()


