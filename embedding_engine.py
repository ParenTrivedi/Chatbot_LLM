
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def generate_embeddings(texts, batch_size=8):
    """
    Generate embeddings for a list of text chunks
    using Nomic through Ollama.
    """

    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch
        }

        try:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json=payload,
                timeout=180
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is installed and running."
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Embedding request timed out."
            )

        except requests.exceptions.HTTPError as error:
            raise RuntimeError(
                f"Ollama embedding error: {error}. "
                "Check whether nomic-embed-text is installed."
            )

        result = response.json()
        batch_embeddings = result.get("embeddings")

        if not batch_embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Number of embeddings does not match "
                "number of text chunks."
            )

        all_embeddings.extend(batch_embeddings)

    return all_embeddings
