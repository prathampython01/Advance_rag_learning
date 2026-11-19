import logging
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import(
    Docx2txtLoader,PyPDFLoader,TextLoader,
)

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

Needed_Extension = {
    '.pdf','.txt','.docx'
}

def load_document(file_path:str | Path) -> List[Document]:

    path = Path(file_path)
    extension = path.suffix.lower() # return the extension of file

    if not path.exists():
        raise FileNotFoundError(f'file not found here:')
    
    if extension not in Needed_Extension:
        raise ValueError(
            f'unsupported file type {extension}.'
        )

    if extension == '.pdf':
        loader = PyPDFLoader(str(path))
    elif extension == '.docx':
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path),encoding='utf-8')

    documents = loader.load()

    new_documents = []

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

        if doc.page_content.strip():
            new_documents.append(doc)

    if not new_documents:
        logger.warning('No readable text found')
        return []

    return add_metadata(new_documents,path)

def load_documents_from_folder(folder_path:str | Path) ->List[Document]:

    folder = Path(folder_path)
    if not folder.exists():
        logger.warning('document folder does not exit')
        return []

    all_documents: List[Document] = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in Needed_Extension:
            continue
        try:
            all_documents.extend(load_document(file_path))
        except Exception as e:
            logger.warning('could not load %s: %s',file_path.name,e)

    logger.info("Loaded %d document pages from %s", len(all_documents), folder)
    return all_documents


def clean_text(text:str)-> str:

    if not text:
        return ''

    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def add_metadata(documents: List[Document],path:Path)-> List[Document]:

    for doc in documents:
        doc.metadata['source'] = str(path)
        doc.metadata['file_name'] = path.name
        doc.metadata['doc_type'] = path.suffix.lower().lstrip('.')

        return documents


def guess_section(text: str) -> str:
    """
    Guess the section heading a piece of text belongs to.

    Policy documents are written as "Password Policy", "Refund Policy" and so
    on. Capturing that heading lets a citation say "Section: Password Policy",
    which is far more useful to a reader than a filename alone.

    Args:
        text: A chunk of document text.

    Returns:
        The heading we found, or an empty string if the text has no obvious
        heading. We deliberately keep this simple: a wrong guess only affects
        the citation label, never the answer itself.
    """
    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()

        # A heading is short, has a handful of words and does not end like a
        # sentence. "4. PASSWORD POLICY" passes; "We reset passwords." fails.
        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue

        # Does this line start like a numbered list item?
        # Examples: "1. Hello", "2) World", "10.3.4 item"
        # It matches: digits, optional ".digits" chunks, optional "." or ")" after the number,
        # then at least one space, then a real word.
        looks_numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line) is not None
        if line.isupper() or line.istitle() or looks_numbered:
            # Strip a leading number so "4. Password Policy" becomes
            # "Password Policy".
            return re.sub(r"^\d+(\.\d+)*[.)]?\s+", "", line).strip()

    return ""
