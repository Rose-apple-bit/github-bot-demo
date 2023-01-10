from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import requests

def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

sources = [
    get_wiki_data("Unix", True),
    get_wiki_data("Microsoft_Windows", True),
    get_wiki_data("Linux", True),
    get_wiki_data("Seinfeld", True),
    get_wiki_data("Matchbox_Twenty", True),
    get_wiki_data("Roman_Empire", True),
    get_wiki_data("London", True),
    get_wiki_data("Python_(programming_language)", True),
    get_wiki_data("Monty_Python", True),
]

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())


chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=3),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
