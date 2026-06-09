from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

from atlassian import Confluence
from slack_sdk import WebClient
from rag.logger import logger

import os


# ✅ helper (centralized logic)
def _apply_default_metadata(doc, filename=None, source=None, classified=False):
    """
    Ensures metadata consistency.
    Missing classification → treated as non-classified.
    """
    if filename:
        doc.metadata["filename"] = filename

    if source:
        doc.metadata["source"] = source

    # ✅ core requirement
    doc.metadata["classified"] = doc.metadata.get("classified", classified)


# ✅ PDF
def load_pdf(file_path, source_name=None, classified=False):
    filename = source_name or os.path.basename(file_path)

    logger.info(
        "Loading PDF %s with source_name=%s classified=%s",
        file_path, filename, classified
    )

    docs = PyPDFLoader(file_path).load()

    for doc in docs:
        # ✅ ensures missing metadata defaults to False
        _apply_default_metadata(doc, filename=filename, classified=classified)

    logger.info("Loaded %s PDF page documents", len(docs))
    return docs


# ✅ Markdown
def load_markdown(file_path, source_name=None, classified=False):
    filename = source_name or os.path.basename(file_path)

    logger.info(
        "Loading markdown %s with source_name=%s classified=%s",
        file_path, filename, classified
    )

    docs = UnstructuredMarkdownLoader(file_path).load()

    for doc in docs:
        _apply_default_metadata(doc, filename=filename, classified=classified)

    logger.info("Loaded %s markdown documents", len(docs))
    return docs


# ✅ Confluence
def load_confluence(classified=False):
    confluence = Confluence(
        url=os.getenv("CONF_URL"),
        username=os.getenv("CONF_USER"),
        password=os.getenv("CONF_TOKEN")
    )

    pages = confluence.get_all_pages_from_space("SPACE", limit=10)

    docs = []
    for p in pages:
        content = confluence.get_page_by_id(
            p["id"],
            expand="body.storage"
        )["body"]["storage"]["value"]

        doc = Document(
            page_content=content,
            metadata={"source": "confluence"}
        )

        # ✅ enforce default behavior
        _apply_default_metadata(doc, source="confluence", classified=classified)

        docs.append(doc)

    return docs


# ✅ Slack
def load_slack(classified=False):
    client = WebClient(token=os.getenv("SLACK_TOKEN"))
    response = client.conversations_history(channel="CHANNEL_ID", limit=50)

    docs = []
    for msg in response["messages"]:
        text = msg.get("text", "")

        doc = Document(
            page_content=text,
            metadata={"source": "slack"}
        )

        # ✅ enforce default behavior
        _apply_default_metadata(doc, source="slack", classified=classified)

        docs.append(doc)

    return docs