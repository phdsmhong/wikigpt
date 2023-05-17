import requests
import wikipedia
from bs4 import BeautifulSoup

wikipedia.set_lang('ko')


def get_wiki(search, numsen):
    lang = "ko"
    """
    fetching summary from wikipedia
    """
    # set language to English (default is auto-detect)
    summary = wikipedia.summary(search, sentences = numsen)

    """
    scrape wikipedia page of the requested query
    """

    # create URL based on user input and language
    url = f"https://{lang}.wikipedia.org/wiki/{search}"

    # send GET request to URL and parse HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # extract main content of page
    content_div = soup.find(id="mw-content-text")

    # extract all paragraphs of content
    paras = content_div.find_all('p')

    # concatenate paragraphs into full page content
    full_page_content = ""
    for para in paras:
        full_page_content += para.text

    # print the full page content
    return full_page_content, summary