import logging
import os
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import psycopg
from config.settings import get_settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from timescale_vector import client


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings and embedding model."""
        self.settings = get_settings()
        
        # Inicializar modelo de embeddings open-source (modo offline)
        try:
            # Intentar modo online primero
            self.embedding_model = SentenceTransformer(self.settings.embedding.model_name)
        except Exception as e:
            print(f"⚠️  Error conectando a Hugging Face: {e}")
            print("🔄 Intentando modo offline...")
            # Forzar modo offline
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.embedding_model = SentenceTransformer(self.settings.embedding.model_name, local_files_only=True)
        
        # Intialize cross-encoder for reranking (open-source)
        try:
            self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"⚠️  Error connecting to cross-encoder: {e}")
            print("🔄 Trying cross-encoder in offline mode...")
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', local_files_only=True)
        
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.settings.embedding.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

    def create_keyword_search_index(self):
        """Create a GIN index for keyword search if it doesn't exist."""
        index_name = f"idx_{self.vector_settings.table_name}_contents_gin"
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {self.vector_settings.table_name} USING gin(to_tsvector('english', contents));
        """
        try:
            with psycopg.connect(self.settings.database.service_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_index_sql)
                    conn.commit()
                    logging.info(f"GIN index '{index_name}' created or already exists.")
        except Exception as e:
            logging.error(f"Error while creating GIN index: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using sentence-transformers.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        
        # Usar sentence-transformers en lugar de OpenAI
        embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
        
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.
        (Renamed from 'search' for compatibility with hybrid search)
        """
        query_embedding = self.get_embedding(query)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        self._log_search_time("Semantic", elapsed_time)

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def keyword_search(
        self, query: str, limit: int = 5, return_dataframe: bool = True
    ) -> Union[List[Tuple[str, str, float]], pd.DataFrame]:
        """
        Perform a keyword search on the contents of the vector store using PostgreSQL full-text search.

        Args:
            query: The search query string.
            limit: The maximum number of results to return. Defaults to 5.
            return_dataframe: Whether to return results as a DataFrame. Defaults to True.

        Returns:
            Either a list of tuples (id, contents, rank) or a pandas DataFrame containing the search results.
        """
        search_sql = f"""
        SELECT id, contents, ts_rank_cd(to_tsvector('english', contents), query) as rank
        FROM {self.vector_settings.table_name}, websearch_to_tsquery('english', %s) query
        WHERE to_tsvector('english', contents) @@ query
        ORDER BY rank DESC
        LIMIT %s
        """

        start_time = time.time()

        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, (query, limit))
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Keyword", elapsed_time)

        if return_dataframe:
            df = pd.DataFrame(results, columns=["id", "content", "rank"])
            df["id"] = df["id"].astype(str)
            return df
        else:
            return results

    def hybrid_search(
        self,
        query: str,
        keyword_k: int = 5,
        semantic_k: int = 5,
        rerank: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Perform a hybrid search combining keyword and semantic search results,
        with optional open-source reranking.

        Args:
            query: The search query string.
            keyword_k: The number of results to return from keyword search.
            semantic_k: The number of results to return from semantic search.
            rerank: Whether to apply cross-encoder reranking (open-source).
            top_n: The number of top results to return after reranking.

        Returns:
            A pandas DataFrame containing the combined search results.
        """
        # Perform keyword search
        keyword_results = self.keyword_search(
            query, limit=keyword_k, return_dataframe=True
        )
        keyword_results["search_type"] = "keyword"
        keyword_results = keyword_results[["id", "content", "search_type"]]

        # Perform semantic search
        semantic_results = self.semantic_search(
            query, limit=semantic_k, return_dataframe=True
        )
        semantic_results["search_type"] = "semantic"
        semantic_results = semantic_results[["id", "content", "search_type"]]

        # Combine results
        combined_results = pd.concat(
            [keyword_results, semantic_results], ignore_index=True
        )

        # Remove duplicates, keeping the first occurrence
        combined_results = combined_results.drop_duplicates(subset=["id"], keep="first")

        if rerank:
            return self._rerank_results_opensource(query, combined_results, top_n)

        return combined_results.head(top_n)

    def _rerank_results_opensource(
        self, query: str, combined_results: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """
        Rerank the combined search results using open-source cross-encoder.

        Args:
            query: The original search query.
            combined_results: DataFrame containing the combined keyword and semantic search results.
            top_n: The number of top results to return after reranking.

        Returns:
            A pandas DataFrame containing the reranked results.
        """
        if len(combined_results) == 0:
            return combined_results

        start_time = time.time()
        
        # Prepare query-document pairs for cross-encoder
        pairs = [[query, content] for content in combined_results["content"].tolist()]
        
        # Get relevance scores using cross-encoder
        scores = self.rerank_model.predict(pairs)
        
        # Add scores to results
        combined_results = combined_results.copy()
        combined_results["relevance_score"] = scores
        
        # Sort by relevance score and return top_n
        reranked_results = combined_results.sort_values(
            "relevance_score", ascending=False
        ).head(top_n)
        
        elapsed_time = time.time() - start_time
        self._log_search_time("Reranking", elapsed_time)
        
        return reranked_results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def _log_search_time(self, search_type: str, elapsed_time: float) -> None:
        """Log the time taken for a search operation."""
        logging.info(f"{search_type} search completed in {elapsed_time:.3f} seconds")

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database."""
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )