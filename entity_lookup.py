import requests
from bs4 import BeautifulSoup

def get_company_info(company):
    search_url = f"https://www.google.com/search?q={company}+official+website"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(search_url, headers=headers)

    soup = BeautifulSoup(res.text, "html.parser")
    links = [
        a["href"] for a in soup.find_all("a", href=True)
        if "http" in a["href"]
    ]

    return {
        "company": company,
        "links": links[:3]
    }
