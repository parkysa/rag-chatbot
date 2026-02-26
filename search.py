from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import numpy as np
from openai import OpenAI

load_dotenv()
DB_PATH = "data/db.parquet"
client = OpenAI()

prompt_template = """
Você é um assistente corporativo.
Seu objetivo é ajudar o usuário de forma clara, natural e amigável, como um atendente humano experiente.
Use apenas as informações da base de conhecimento.
Responda apenas o que foi perguntado. Não adicione explicações extras, contexto adicional ou informações relacionadas que não foram solicitadas.
Não faça respostas longas.
Se a resposta não estiver na base de conhecimento, diga: "Não encontrei essa informação na base de conhecimento disponível."

Pergunta:
{pergunta}

Base de conhecimento:
{base_conhecimento}
"""

def cosine_similarity(a, b): # a é embedding da pergunta e b embedding do chunk
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) # obtem o consseno do angulo

def search():
    question = input("Escreva sua pergunta: ")

    embedding_function = OpenAIEmbeddings()
    question_embedding = embedding_function.embed_query(question) # vetorizei a pergunta

    db = pd.read_parquet(DB_PATH)

    db["similarity"] = db["embedding"].apply( #  para cada embedding no banco execute essa função
        lambda x: cosine_similarity(question_embedding, np.array(x)) # calcula a similaridade e coloca na coluna similarity
    )

    # ordena a similaridade do maior pro menor e pega os 3 primeiros
    top_k = db.sort_values("similarity", ascending=False).head(3)

    # pego o texto dos 3 chunks e coloco numa string com separador
    base_conhecimento = "\n\n".join(top_k["texto"].tolist())

    final_prompt = prompt_template.format(
        pergunta=question,
        base_conhecimento=base_conhecimento
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente corporativo."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0
    )

    print("\nResposta:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    search()