import pathlib
import subprocess
import tempfile
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.docstore.document import Document

max_files = 100

def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("*/*.py")) + list(
            repo_path.glob("*/*.py")
        )
        for markdown_file in markdown_files:
            if max_files <= 0:
                break
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                print(f"Processing {github_url}")
                yield Document(page_content=f.read(), metadata={"source": github_url})
                --max_files


sources = get_github_docs("dagster-io", "dagster")

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

