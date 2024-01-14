from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate

import json
import os
from dotenv import dotenv_values
config = dotenv_values(".env")

with open(f"few_shot_prompts.json", "r") as f:
    few_shots = json.load(f)["few_shots"]

db_user = config["DB_USER"]
db_password = config["DB_PASS"]
db_host = config["DB_HOST"]
db_name = config["DB_NAME"]

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                            sample_rows_in_table_info=3)
llm = GooglePalm(google_api_key=config["PALM_API_KEY"], temperature=0.1)
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
to_vectorize = [" ".join(example.values()) for example in few_shots]
vectorstore = Chroma.from_texts(to_vectorize, embedding_function, metadatas=few_shots)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
)

print(PROMPT_SUFFIX)

def get_few_shot_db_chain():
    """Function to prepare LLM chain that accepts as input a question, and then compares the question against all embeddings in Chroma Vector Store, and uses the 2 most similar examples for few shot example generation. Final result generated using RAGs.

    Returns:
        langchain chain: lang chain chain to perform RAG to generate few shot prompt
    """    
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain
