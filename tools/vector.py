import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA
from llm import llm, embeddings

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
    index_name="articleBody",
    node_label="Article",
    text_node_property="body",
    embedding_node_property="embedding",
    retrieval_query="""
RETURN
    node.body AS text,
    score,
    {
	id: node.id,
        title: node.title,
        authors: [ (author)-[:AUTHORED]->(node) | author.title ],
	tickers: [ (node)-[:ABOUT_TICKER]->(ticker) | ticker.symbol ],
	url: node.url
    } AS metadata
""",
)

retriever = neo4jvector.as_retriever()


kg_qa = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
)
