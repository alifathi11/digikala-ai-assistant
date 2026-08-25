import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self):

        self.index = None
        self.documents = None


    def build(
        self,
        embeddings,
        documents
    ):

        embeddings = np.asarray(
            embeddings
        ).astype("float32")


        dimension = embeddings.shape[1]


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.documents = documents.reset_index(
            drop=True
        )


    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.asarray(
            query_embedding
        ).astype("float32")


        scores, indices = self.index.search(
            query_embedding,
            top_k
        )


        results = self.documents.iloc[
            indices[0]
        ].copy()


        results["score"] = scores[0]


        return results