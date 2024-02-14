import requests
from bs4 import BeautifulSoup

url = "https://news.naver.com/main/ranking/popularMemo.naver"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, 'lxml')

newslist = soup.select(".rankingnews_list")
len(newslist)