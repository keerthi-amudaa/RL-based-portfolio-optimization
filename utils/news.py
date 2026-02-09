from newsapi import NewsApiClient

def fetch_news(ticker, api_key, page_size=5):
    # 🔒 Force page_size to be int (fixes error)
    page_size = int(page_size)

    newsapi = NewsApiClient(api_key=api_key)

    response = newsapi.get_everything(
        q=ticker,
        language="en",
        sort_by="relevancy",
        page_size=page_size
    )

    return response.get("articles", [])
