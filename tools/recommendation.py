import json
import streamlit as st
from openai import OpenAI
from graph import graph

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

system_prompt = f"""
    You are a helpful agent to provide recommendation for readers of the specific article.

    Depending on the user prompt, determine if it possible to answer with the graph database.

    The graph database can match products with multiple relationships to several entities.

    Example user input:
    "Recommend other articles for readers of the article titled 'This is Foo'"

    For the example provided, the expected output would be:
    {{
        "title": "This is Foo",
    }}

    If there are no relevant information in the user prompt, return an empty json object.
"""


# Define the entities to look for
def define_query(prompt, model="gpt-4-1106-preview"):
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def create_query(text, threshold=0.81):
    query_data = json.loads(text)
    query = f"""
        MATCH (m:Article {{title: '{query_data['title']}'}})
        CALL db.index.vector.queryNodes('articleBody', 6, m.embedding)
        YIELD node, score
	WHERE score <> 1
        RETURN node.title AS title, node.url AS url, score
    """
    return query


def query_graph(response):
    query = create_query(response)
    result = graph.query(query)
    return result


def run(prompt):
    if type(prompt) == str:
        params = json.dumps({'title': prompt})
    else:
        params = define_query(prompt)
    result = query_graph(params)
    if not result:
        return "I can't find any recommendations."

    output = ""
    for i, r in enumerate(result):
        output += f"{i+1}. {r['title']}\n{r['url']}\n"

    return output
