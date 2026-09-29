import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    documents_dir = "resumes"
    documents, metadatas, ids = [], [], []
    id = 0

    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(documents_dir, filename), "r", encoding="utf-8") as file:
                chunks = file.read().replace("\n", ".").split("### ")
                for chunk in chunks:
                    if chunk and not chunk.isspace():
                        documents.append(chunk)
                        metadatas.append({"source": filename, "info": chunks[1]})
                        ids.append(str(id))
                        id += 1

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="CVs",
        embedding_function=openai_ef
    )
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    user_question = "mi serve qualcuno per promuovere il mio prodotto"

    results = collection.query(
        query_texts=[user_question],
        n_results=1
    )

    best = results["metadatas"][0][0]
    context = (
        f"CONTESTO: nome file {best['source']}. "
        f"Paragrafo più significativo: {results['documents'][0][0]}. "
        f"Menziona il nome del candidato all'inizio e i dati personali alla fine per il contatto: {best['info']}"
    )

    prompt = f"""Dato il seguente contesto {context} rispondi alla domanda dell'utente {user_question}
spiegando che nel file individuato c'è il profilo più adatto.
Argomenta la scelta usando il contenuto del testo individuato nel contesto."""

    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente HR, specializzato nella ricerca di profili professionali"},
            {"role": "user", "content": prompt},
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()