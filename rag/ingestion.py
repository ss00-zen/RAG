from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

from atlassian import Confluence
from slack_sdk import WebClient

import os

def load_pdf(file_path):
    return PyPDFLoader(file_path).load()

def load_markdown(file_path):
    return UnstructuredMarkdownLoader(file_path).load()

# ✅ Confluence (real API)
def load_confluence():
    confluence = Confluence(
        url=os.getenv("CONF_URL"),
        username=os.getenv("CONF_USER"),
        password=os.getenv("CONF_TOKEN")
    )

    pages = confluence.get_all_pages_from_space("SPACE", limit=10)

    docs = []
    for p in pages:
        content = confluence.get_page_by_id(p["id"], expand="body.storage")["body"]["storage"]["value"]
        docs.append(Document(page_content=content, metadata={"source": "confluence"}))

    return docs

# ✅ Slack (real)
def load_slack():
    client = WebClient(token=os.getenv("SLACK_TOKEN"))
    response = client.conversations_history(channel="CHANNEL_ID", limit=50)

    docs = []
    for msg in response["messages"]:
        docs.append(Document(page_content=msg["text"], metadata={"source": "slack"}))

    return docs
