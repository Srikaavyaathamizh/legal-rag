"""
Phase 4 — Semantic Search
-------------------------

SentenceTransformer embeddings + Qdrant vector database.

Supports:

1. In-memory Qdrant
   ":memory:"

2. Persistent local Qdrant
   "./qdrant_data"

3. Remote Qdrant server
   "http://localhost:6333"

The local persistent mode is recommended for this project because
the semantic embeddings do not need to be rebuilt every time the
FastAPI server restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from temporal_filter import (
    get_candidate_sections,
    LawResolution,
)


# ================================================================
# CONFIGURATION
# ================================================================

COLLECTION_NAME = "legal_sections"

MODEL_NAME = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 produces 384-dimensional vectors.
VECTOR_SIZE = 384

# Persistent local Qdrant database.
DEFAULT_QDRANT_LOCATION = "./qdrant_data"


# ================================================================
# RESULT
# ================================================================

@dataclass
class SemanticResult:
    section_record: dict
    score: float


# ================================================================
# SEMANTIC INDEX
# ================================================================

class SemanticIndex:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        qdrant_location: str = DEFAULT_QDRANT_LOCATION,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Create SentenceTransformer + Qdrant connection.

        qdrant_location can be:

            ":memory:"
                Temporary in-memory database.

            "./qdrant_data"
                Persistent local Qdrant database.

            "C:/some/path/qdrant_data"
                Windows persistent path.

            "http://localhost:6333"
                Remote/local Qdrant server.
        """

        # ------------------------------------------------------------
        # Load embedding model
        # ------------------------------------------------------------

        print("=" * 70)
        print("Loading SentenceTransformer model...")
        print(f"Model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("SentenceTransformer loaded successfully.")
        print(f"Embedding dimension: {VECTOR_SIZE}")

        # ------------------------------------------------------------
        # Store configuration
        # ------------------------------------------------------------

        self.collection_name = collection_name

        # Maps Qdrant point ID -> complete legal section record.
        self._section_lookup: dict[int, dict] = {}

        # ------------------------------------------------------------
        # Create Qdrant client
        # ------------------------------------------------------------

        self.client = self._create_qdrant_client(
            qdrant_location
        )

        print(
            f"Qdrant collection: {self.collection_name}"
        )

    # ================================================================
    # QDRANT CLIENT
    # ================================================================

    def _create_qdrant_client(
        self,
        qdrant_location: str,
    ) -> QdrantClient:
        """
        Creates a Qdrant client safely.

        IMPORTANT:
        On Windows, a path such as

            C:\\Users\\admin\\Music\\legal_rag\\qdrant_data

        must be passed using:

            QdrantClient(path=...)

        NOT:

            QdrantClient(location=...)

        Otherwise Qdrant may interpret "C:" as a URL scheme and
        produce:

            ValueError: Unknown scheme: c
        """

        # ------------------------------------------------------------
        # In-memory Qdrant
        # ------------------------------------------------------------

        if qdrant_location == ":memory:":
            print(
                "Qdrant mode: IN-MEMORY"
            )

            return QdrantClient(
                location=":memory:"
            )

        # ------------------------------------------------------------
        # Remote Qdrant
        # ------------------------------------------------------------

        if qdrant_location.startswith(
            ("http://", "https://")
        ):
            print(
                f"Qdrant mode: REMOTE\n"
                f"Qdrant URL: {qdrant_location}"
            )

            return QdrantClient(
                url=qdrant_location
            )

        # ------------------------------------------------------------
        # Persistent local Qdrant
        # ------------------------------------------------------------

        qdrant_path = Path(
            qdrant_location
        ).resolve()

        qdrant_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "Qdrant mode: PERSISTENT LOCAL"
        )

        print(
            f"Qdrant path: {qdrant_path}"
        )

        return QdrantClient(
            path=str(qdrant_path)
        )

    # ================================================================
    # BUILD INDEX
    # ================================================================

    def build(
        self,
        all_sections: list[dict],
        batch_size: int = 64,
        force_rebuild: bool = False,
    ) -> None:
        """
        Build the Qdrant semantic index.

        Parameters
        ----------
        all_sections:
            Complete legal corpus.

        batch_size:
            Number of documents embedded per batch.

        force_rebuild:
            False:
                Reuse an existing collection.

            True:
                Delete and recreate the collection.
        """

        # ------------------------------------------------------------
        # Remove internally repealed IPC records
        # ------------------------------------------------------------

        records = [
            section
            for section in all_sections
            if section["status"] != "repealed_in_ipc"
        ]

        print(
            f"Corpus sections available: {len(records)}"
        )

        # ------------------------------------------------------------
        # Force rebuild
        # ------------------------------------------------------------

        if force_rebuild:

            if self.client.collection_exists(
                self.collection_name
            ):

                print(
                    f"Deleting existing collection "
                    f"'{self.collection_name}'..."
                )

                self.client.delete_collection(
                    self.collection_name
                )

        # ------------------------------------------------------------
        # Existing collection
        # ------------------------------------------------------------

        if self.client.collection_exists(
            self.collection_name
        ):

            print("=" * 70)
            print(
                f"Qdrant collection "
                f"'{self.collection_name}' already exists."
            )

            print(
                "Reusing existing semantic index."
            )

            # Reconstruct the ID -> record lookup.
            #
            # Point IDs are deterministic because they were created
            # using enumerate(records).
            self._section_lookup = {
                idx: record
                for idx, record in enumerate(records)
            }

            print(
                f"Loaded {len(self._section_lookup)} "
                f"section mappings."
            )

            print("=" * 70)

            return

        # ------------------------------------------------------------
        # Create collection
        # ------------------------------------------------------------

        print("=" * 70)

        print(
            f"Creating Qdrant collection "
            f"'{self.collection_name}'..."
        )

        self.client.create_collection(
            collection_name=self.collection_name,

            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            "Qdrant collection created."
        )

        # ------------------------------------------------------------
        # Prepare texts
        # ------------------------------------------------------------

        texts = [
            f"{record['heading']} {record['text']}"
            for record in records
        ]

        print(
            f"Generating embeddings for "
            f"{len(texts)} sections..."
        )

        # ------------------------------------------------------------
        # Generate embeddings
        # ------------------------------------------------------------

        embeddings = self.model.encode(
            texts,

            batch_size=batch_size,

            show_progress_bar=True,

            normalize_embeddings=True,
        )

        print(
            "Embedding generation completed."
        )

        # ------------------------------------------------------------
        # Create points
        # ------------------------------------------------------------

        points = []

        for idx, (record, vector) in enumerate(
            zip(records, embeddings)
        ):

            self._section_lookup[idx] = record

            point = PointStruct(
                id=idx,

                vector=vector.tolist(),

                payload={
                    "law": record["law"],
                    "section": record["section"],
                    "status": record["status"],
                    "heading": record["heading"],
                },
            )

            points.append(point)

        # ------------------------------------------------------------
        # Upsert
        # ------------------------------------------------------------

        print(
            f"Upserting {len(points)} vectors into Qdrant..."
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            "Semantic Qdrant index built successfully."
        )

        print("=" * 70)

    # ================================================================
    # LOAD SECTION LOOKUP
    # ================================================================

    def load_section_lookup(
        self,
        all_sections: list[dict],
    ) -> None:
        """
        Reconstruct the deterministic Qdrant point-ID -> section mapping.

        Qdrant point IDs are assigned using enumerate(records), where
        records excludes IPC sections already repealed internally.

        This is required when a persistent Qdrant collection is opened
        in a fresh Python process without calling build().
        """

        records = [
            section
            for section in all_sections
            if section["status"] != "repealed_in_ipc"
        ]

        self._section_lookup = {
            idx: record
            for idx, record in enumerate(records)
        }

        print(
            f"Loaded {len(self._section_lookup)} "
            f"section mappings."
        )

    # ================================================================
    # SEARCH
    # ================================================================

    def search(
        self,
        query: str,
        law: Optional[str] = None,
        candidate_section_ids: Optional[set[str]] = None,
        top_k: int = 10,
    ) -> list[SemanticResult]:
        """
        Semantic search with optional Qdrant metadata filters.

        Examples:

            search(
                "death by negligent driving",
                law="IPC"
            )

        Or:

            search(
                "death by negligent driving",
                law="IPC",
                candidate_section_ids={"304A", "279"}
            )
        """

        if not query or not query.strip():
            return []

        # ------------------------------------------------------------
        # Build filters
        # ------------------------------------------------------------

        must_conditions = []

        # Law filter
        if law is not None:

            must_conditions.append(
                FieldCondition(
                    key="law",

                    match=MatchValue(
                        value=law
                    ),
                )
            )

        # Candidate section filter
        if candidate_section_ids is not None:

            # If candidate set is empty, immediately return.
            if not candidate_section_ids:
                return []

            must_conditions.append(
                FieldCondition(
                    key="section",

                    match=MatchAny(
                        any=list(candidate_section_ids)
                    ),
                )
            )

        # ------------------------------------------------------------
        # Qdrant filter
        # ------------------------------------------------------------

        query_filter = (
            Filter(
                must=must_conditions
            )
            if must_conditions
            else None
        )

        # ------------------------------------------------------------
        # Encode query
        # ------------------------------------------------------------

        query_vector = self.model.encode(
            query,

            normalize_embeddings=True,
        ).tolist()

        # ------------------------------------------------------------
        # Query Qdrant
        # ------------------------------------------------------------

        response = self.client.query_points(
            collection_name=self.collection_name,

            query=query_vector,

            query_filter=query_filter,

            limit=top_k,
        )

        # ------------------------------------------------------------
        # Convert Qdrant results
        # ------------------------------------------------------------

        results = []

        for hit in response.points:

            point_id = int(hit.id)

            if point_id not in self._section_lookup:

                raise RuntimeError(
                    f"Qdrant returned unknown point ID "
                    f"{point_id}. "
                    "The persistent index and corpus are "
                    "out of sync. Rebuild the semantic index."
                )

            record = self._section_lookup[
                point_id
            ]

            results.append(
                SemanticResult(
                    section_record=record,
                    score=float(hit.score),
                )
            )

        return results

    # ================================================================
    # COLLECTION INFO
    # ================================================================

    def collection_exists(self) -> bool:
        """
        Returns True if the Qdrant collection exists.
        """

        return self.client.collection_exists(
            self.collection_name
        )

    # ================================================================
    # DELETE COLLECTION
    # ================================================================

    def delete_collection(self) -> None:
        """
        Delete the current semantic collection.

        Use this when you intentionally want to rebuild embeddings.
        """

        if self.client.collection_exists(
            self.collection_name
        ):

            print(
                f"Deleting collection "
                f"'{self.collection_name}'..."
            )

            self.client.delete_collection(
                self.collection_name
            )

            self._section_lookup.clear()

            print(
                "Collection deleted."
            )


# ====================================================================
# TEMPORAL + SEMANTIC SEARCH
# ====================================================================

def search_with_temporal_filter(
    index: "SemanticIndex",
    all_sections: list[dict],
    query: str,
    incident_date,
    top_k: int = 10,
) -> tuple[
    list[SemanticResult],
    LawResolution,
]:
    """
    Phase 2 + Phase 4 integration.

    Steps:

        1. Resolve IPC/BNS from incident date.
        2. Get legal candidate sections.
        3. Search only the applicable law.
        4. Apply candidate-section filtering inside Qdrant.

    If no date is available, search both IPC and BNS.
    """

    # ------------------------------------------------------------
    # Restore persistent Qdrant ID -> corpus mapping
    # ------------------------------------------------------------
    #
    # A persistent Qdrant collection survives Python restarts,
    # but _section_lookup exists only in memory.
    #
    # Reconstruct it from the same corpus ordering used by build().
    #

    if index.collection_exists() and not index._section_lookup:
        index.load_section_lookup(
            all_sections
        )

    # ------------------------------------------------------------
    # Temporal resolution
    # ------------------------------------------------------------

    candidates, resolution = get_candidate_sections(
        all_sections,
        incident_date,
    )

    # ------------------------------------------------------------
    # Candidate section IDs
    # ------------------------------------------------------------

    candidate_ids = {
        section["section"]
        for section in candidates
    }

    # ------------------------------------------------------------
    # Ambiguous date
    # ------------------------------------------------------------

    if resolution.law == "AMBIGUOUS":

        ipc_results = index.search(
            query=query,

            law="IPC",

            candidate_section_ids=candidate_ids,

            top_k=top_k,
        )

        bns_results = index.search(
            query=query,

            law="BNS",

            candidate_section_ids=candidate_ids,

            top_k=top_k,
        )

        # --------------------------------------------------------
        # Interleave IPC/BNS results
        # --------------------------------------------------------

        interleaved = []

        for ipc_result, bns_result in zip(
            ipc_results,
            bns_results,
        ):

            interleaved.append(
                ipc_result
            )

            interleaved.append(
                bns_result
            )

        # Add remaining results.
        interleaved.extend(
            ipc_results[
                len(bns_results):
            ]
        )

        interleaved.extend(
            bns_results[
                len(ipc_results):
            ]
        )

        return (
            interleaved,
            resolution,
        )

    # ------------------------------------------------------------
    # Known law
    # ------------------------------------------------------------

    results = index.search(
        query=query,

        law=resolution.law,

        candidate_section_ids=candidate_ids,

        top_k=top_k,
    )

    return (
        results,
        resolution,
    )


# ====================================================================
# LOAD CORPUS
# ====================================================================

def load_corpus(
    path: str = "data/processed/unified_sections.json",
) -> list[dict]:
    """
    Load unified legal corpus.

    UTF-8 is explicitly specified because Windows otherwise defaults
    to cp1252, which previously caused:

        UnicodeDecodeError:
        'charmap' codec can't decode byte ...
    """

    corpus_path = Path(path).resolve()

    if not corpus_path.exists():

        raise FileNotFoundError(
            f"Corpus file not found:\n{corpus_path}"
        )

    with open(
        corpus_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if "sections" not in data:

        raise ValueError(
            "Invalid corpus format. "
            "Expected top-level 'sections' key."
        )

    return data["sections"]