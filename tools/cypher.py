from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

from llm import llm
from graph import graph

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about Tempest and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statements:

1. When asked to generate key points from the article titled 'This is Foo', you can use this query to load the content, then generate key points from it.
```
MATCH (a:Article {{title: "This is Foo"}})
RETURN a.body as article
```

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)
# cypher_qa = GraphCypherQAChain.from_llm(
#     llm=llm,
#     graph=graph,  # (2)
#     verbose=True
# )

# response = cypher_qa.run("List title and url of five oldest articles about TSLA.")

# print(response)
