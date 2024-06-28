### TEST RESULTS

- **Project Top:** [主目录](https://github.com/Jeremyywb/TextCompete/blob/main/exam)
- **Reformatted schema:** 

```json
{
  "id": "<rowid+segment id>",
  "head":"<headline text>",
  "Question": "<question_text>",
  "Answer": "<answer_text>"
}
```
- **Transformation Code:**  [code](https://github.com/Jeremyywb/TextCompete/blob/main/exam/ProcessingCode.py)
- **Output File:** [output](https://github.com/Jeremyywb/TextCompete/blob/main/exam/AdaptLLM-finance-tasks-Headline.json)
- **Statistics:** 102735 question-answer pairs
- **Performance Metrics:** 1.4s for cleanning process

### 附带信息

- **姓名:** 游文斌
- **意向岗位:** 大模型数据工程师
- **学校:** 福州大学
- **学历:** 本科
- **实习开始时间:** 社会招聘
- **可持续时间:** 社会招聘
- **可否在杭州办公:** 是
- **期望从实习中收获什么:** 社会招聘



import requests
from bs4 import BeautifulSoup

def get_latest_news():
    url = 'https://www.forkliftaction.com/news/news.aspx'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.find_all('div', class_='newsitem')
        
        for item in news_items:
            title = item.find('h2').text.strip()
            summary = item.find('p').text.strip()
            print(f"Title: {title}\nSummary: {summary}\n")
    else:
        print(f"Failed to retrieve news. Status code: {response.status_code}")

if __name__ == '__main__':
    get_latest_news()


