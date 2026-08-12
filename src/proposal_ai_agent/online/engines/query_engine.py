"""Query reception engine for the online pipeline."""

from __future__ import annotations

from hashlib import sha256
from math import ceil
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
import unicodedata
from uuid import uuid4

from ..benchmarks import BenchmarkProfile, BenchmarkRegistry
from ..benchmarks.profiles import RAG_QA_PROFILE
from ..contracts import (
    ClarificationRequest,
    ProcessedQuery,
    QueryEmbedding,
    QualifiedQuery,
    UserQuery,
    ValidationResult,
)
from ..contracts import (
    ACLPolicy,
    HybridSearchPolicy,
    RerankingPolicy,
    RetrievalBudget,
    RetrievalRequest,
    RetrievalStrategy,
    SearchScope,
)
from ..providers.embedding import EmbeddingCache, EmbeddingProvider, MemoryCache
from proposal_ai_agent.embeddings.validators import VectorValidator


class QueryEngine:
    """Normalize raw user input into the immutable online query contract."""

    MAX_QUERY_LENGTH = 10_000
    PROCESSING_VERSION = "1.0"
    PLANNER_VERSION = "1.0"
    STRATEGY_VERSION = "1.0"
    DEFAULT_TOP_K = 5
    DEFAULT_CANDIDATE_BUDGET = 20
    DEFAULT_SCORE_THRESHOLD = 0.0
    DEFAULT_MAX_LATENCY_MS = 1_000
    DEFAULT_MAX_CONTEXT_TOKENS = 4_000

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        default_language: str = "en",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_dimension: int | None = None,
        embedding_cache: EmbeddingCache | None = None,
        embedding_model_id: str | None = None,
    ) -> None:
        if not default_language.strip():
            raise ValueError("default_language must not be empty")
        self._registry = registry or BenchmarkRegistry((RAG_QA_PROFILE,))
        self._default_language = default_language.strip()
        if embedding_provider is None:
            if embedding_dimension is not None or embedding_cache is not None or embedding_model_id is not None:
                raise ValueError("embedding configuration requires an embedding_provider")
            self._embedding_provider = None
            self._embedding_cache = None
            self._embedding_validator = None
            self._embedding_model_id = None
        else:
            if embedding_dimension is None:
                raise ValueError("embedding_dimension is required with an embedding_provider")
            if not embedding_model_id and not embedding_provider.__class__.__name__:
                raise ValueError("embedding_model_id must not be empty")
            self._embedding_provider = embedding_provider
            self._embedding_cache = embedding_cache or MemoryCache()
            self._embedding_validator = VectorValidator(embedding_dimension)
            self._embedding_model_id = embedding_model_id or embedding_provider.__class__.__name__

    def receive_query(
        self,
        query: str,
        session_id: str | None = None,
        conversation_history: Sequence[Mapping[str, Any]] | None = None,
        user_context: Mapping[str, Any] | None = None,
        auth_context: Mapping[str, Any] | None = None,
    ) -> UserQuery:
        """Receive one query without invoking downstream pipeline behavior."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        return UserQuery(
            request_id=uuid4(),
            session_id=session_id,
            query=query.strip(),
            timestamp_utc=datetime.now(timezone.utc),
            conversation_history=tuple(conversation_history or ()),
            user_context=user_context,
            auth_context=auth_context,
            trace_metadata={},
        )

    def qualify_query(self, user_query: UserQuery) -> QualifiedQuery:
        """Qualify a received query against its selected benchmark profile."""
        if not isinstance(user_query, UserQuery):
            raise TypeError("user_query must be a UserQuery")

        profile = self._registry.select(user_query.query)
        parameters, ambiguity_flags, conflict_flags = self._extract_parameters(
            user_query.query, profile
        )
        for name, value in profile.defaults.items():
            parameters.setdefault(name, value)

        missing_parameters = tuple(
            name for name in profile.required_parameters if not parameters.get(name)
        )
        optional_parameters = tuple(
            name for name in profile.optional_parameters if name in parameters
        )
        validation_result = self._validate(parameters, profile, missing_parameters)
        confidence_score = self._confidence(
            missing_parameters, ambiguity_flags, conflict_flags, validation_result
        )
        clarification_requests = self._clarifications(
            missing_parameters,
            ambiguity_flags,
            conflict_flags,
            confidence_score,
            profile,
        )

        return QualifiedQuery(
            original=user_query,
            intent=profile.intent_id,
            benchmark_id=profile.intent_id,
            extracted_parameters=parameters,
            missing_parameters=missing_parameters,
            optional_parameters=optional_parameters,
            confidence_score=confidence_score,
            ambiguity_flags=ambiguity_flags,
            conflict_flags=conflict_flags,
            clarification_required=bool(clarification_requests),
            clarification_requests=clarification_requests,
            validation_result=validation_result,
        )

    def process_query(self, qualified_query: QualifiedQuery) -> ProcessedQuery:
        """Validate and syntactically prepare a qualified query without requalifying it."""
        if not isinstance(qualified_query, QualifiedQuery):
            raise TypeError("qualified_query must be a QualifiedQuery")

        original_query = qualified_query.original.query
        self._validate_query_text(original_query)
        normalized_query = self._normalize_query_text(original_query)
        language, language_confidence = self._detect_language(normalized_query)
        query_hash = sha256(normalized_query.encode("utf-8")).hexdigest()
        unicode_normalized = unicodedata.normalize("NFC", original_query)
        flags = {
            "encoding_valid": True,
            "unicode_normalized": unicode_normalized != original_query,
            "line_breaks_normalized": "\r" in original_query,
            "whitespace_normalized": normalized_query != original_query,
        }

        return ProcessedQuery(
            qualified_query=qualified_query,
            normalized_query=normalized_query,
            language=language,
            language_confidence=language_confidence,
            query_hash=query_hash,
            cache_key=f"query:{query_hash}",
            character_count=len(normalized_query),
            word_count=len(normalized_query.split()),
            estimated_token_count=ceil(len(normalized_query) / 4),
            processing_timestamp_utc=datetime.now(timezone.utc),
            processing_version=self.PROCESSING_VERSION,
            processing_flags=flags,
        )

    def embed_query(self, processed_query: ProcessedQuery) -> QueryEmbedding:
        """Embed one processed query using the shared cache-aware provider contract."""
        if not isinstance(processed_query, ProcessedQuery):
            raise TypeError("processed_query must be a ProcessedQuery")
        return self._embed_processed_queries((processed_query,))[0]

    def plan_retrieval(self, query_embedding: QueryEmbedding) -> RetrievalRequest:
        """Create a deterministic retrieval blueprint without executing retrieval."""
        if not isinstance(query_embedding, QueryEmbedding):
            raise TypeError("query_embedding must be a QueryEmbedding")

        qualified_query = query_embedding.processed_query.qualified_query
        profile = self._registry.get(qualified_query.benchmark_id)
        defaults = profile.defaults
        parameters = qualified_query.extracted_parameters
        metadata_filters = {
            name: value for name, value in parameters.items() if name != "question"
        }
        top_k = self._integer_default(defaults, "top_k", self.DEFAULT_TOP_K)
        candidate_budget = self._integer_default(
            defaults, "candidate_budget", self.DEFAULT_CANDIDATE_BUDGET
        )
        budget = RetrievalBudget(
            max_candidates=self._integer_default(
                defaults, "max_candidates", candidate_budget
            ),
            max_latency_ms=self._integer_default(
                defaults, "max_latency_ms", self.DEFAULT_MAX_LATENCY_MS
            ),
            max_context_tokens=self._integer_default(
                defaults, "max_context_tokens", self.DEFAULT_MAX_CONTEXT_TOKENS
            ),
        )
        strategy = self._enum_default(
            RetrievalStrategy, defaults, "retrieval_strategy", RetrievalStrategy.DENSE
        )

        return RetrievalRequest(
            query_embedding=query_embedding,
            metadata_filters=metadata_filters,
            search_scope=SearchScope(
                knowledge_base=defaults.get("knowledge_base"),
                collection=defaults.get("collection"),
                document=parameters.get("document"),
                version=parameters.get("version"),
                extensions={
                    name: value
                    for name, value in parameters.items()
                    if name not in {"question", "document", "version"}
                },
            ),
            retrieval_strategy=strategy,
            retrieval_profile=defaults.get("retrieval_profile", profile.intent_id),
            top_k=top_k,
            candidate_budget=candidate_budget,
            score_threshold=self._float_default(
                defaults, "score_threshold", self.DEFAULT_SCORE_THRESHOLD
            ),
            reranking_policy=self._enum_default(
                RerankingPolicy, defaults, "reranking_policy", RerankingPolicy.DISABLED
            ),
            hybrid_policy=self._enum_default(
                HybridSearchPolicy, defaults, "hybrid_policy", HybridSearchPolicy.DENSE_ONLY
            ),
            acl_policy=ACLPolicy.PLACEHOLDER,
            retrieval_budget=budget,
            planner_metadata={
                "planner_version": self.PLANNER_VERSION,
                "benchmark_id": profile.intent_id,
                "strategy_version": self.STRATEGY_VERSION,
            },
        )

    @staticmethod
    def _integer_default(defaults: Mapping[str, str], name: str, fallback: int) -> int:
        """Resolve one integer retrieval setting from profile defaults."""
        return int(defaults.get(name, fallback))

    @staticmethod
    def _float_default(defaults: Mapping[str, str], name: str, fallback: float) -> float:
        """Resolve one float retrieval setting from profile defaults."""
        return float(defaults.get(name, fallback))

    @staticmethod
    def _enum_default(
        enum_type: type[Any],
        defaults: Mapping[str, str],
        name: str,
        fallback: Any,
    ) -> Any:
        """Resolve one enum retrieval setting from profile defaults."""
        return enum_type(defaults.get(name, fallback.value))

    def _embed_processed_queries(
        self, processed_queries: Sequence[ProcessedQuery]
    ) -> tuple[QueryEmbedding, ...]:
        """Batch-ready embedding workflow that invokes the provider only for cache misses."""
        if self._embedding_provider is None or self._embedding_cache is None or self._embedding_validator is None:
            raise RuntimeError("QueryEngine has no embedding provider configured")

        ordered_queries = tuple(processed_queries)
        vectors: dict[str, tuple[float, ...]] = {}
        missed_keys: set[str] = set()
        missing_queries: dict[str, ProcessedQuery] = {}
        for processed_query in ordered_queries:
            cache_key = processed_query.cache_key
            if cache_key in vectors or cache_key in missing_queries:
                continue
            cached_vector = self._embedding_cache.get(cache_key)
            if cached_vector is None:
                missing_queries[cache_key] = processed_query
                missed_keys.add(cache_key)
            else:
                vectors[cache_key] = self._embedding_validator.validate(cached_vector)

        if missing_queries:
            missing_items = tuple(missing_queries.items())
            generated_vectors = self._embedding_provider.embed_batch(
                tuple(item.normalized_query for _, item in missing_items)
            )
            if len(generated_vectors) != len(missing_items):
                raise ValueError("provider returned a vector count different from request count")
            for (cache_key, _), vector in zip(missing_items, generated_vectors):
                validated_vector = self._embedding_validator.validate(vector)
                self._embedding_cache.set(cache_key, validated_vector)
                vectors[cache_key] = validated_vector

        return tuple(
            QueryEmbedding(
                processed_query=processed_query,
                vector=vectors[processed_query.cache_key],
                model_id=self._embedding_model_id or "unknown",
                embedding_dimension=self._embedding_validator.dimensions,
                cache_hit=processed_query.cache_key not in missed_keys,
                embedding_timestamp_utc=datetime.now(timezone.utc),
                embedding_metadata={
                    "cache_key": processed_query.cache_key,
                    "processing_version": processed_query.processing_version,
                },
            )
            for processed_query in ordered_queries
        )

    def _validate_query_text(self, query: str) -> None:
        """Reject text that cannot be safely handled by downstream text systems."""
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters")
        try:
            query.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("query must be valid UTF-8 text") from error
        if any(
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\r", "\t"}
            for character in query
        ):
            raise ValueError("query contains unsupported control characters")

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        """Apply semantic-preserving Unicode, line-break, and whitespace normalization."""
        unicode_normalized = unicodedata.normalize("NFC", query)
        normalized_line_breaks = unicode_normalized.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\s+", " ", normalized_line_breaks).strip()

    def _detect_language(self, query: str) -> tuple[str, float]:
        """Use Unicode script evidence, falling back to the configured language."""
        script_ranges = {
            "ar": ((0x0600, 0x06FF),),
            "hi": ((0x0900, 0x097F),),
            "ru": ((0x0400, 0x052F),),
            "zh": ((0x4E00, 0x9FFF),),
            "ja": ((0x3040, 0x30FF),),
            "ko": ((0xAC00, 0xD7AF),),
        }
        letters = [character for character in query if character.isalpha()]
        if not letters:
            return self._default_language, 0.0
        for language, ranges in script_ranges.items():
            matches = sum(
                any(start <= ord(character) <= end for start, end in ranges)
                for character in letters
            )
            if matches:
                return language, round(matches / len(letters), 2)
        return self._default_language, 0.6

    @staticmethod
    def _extract_parameters(
        query: str, profile: BenchmarkProfile
    ) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
        """Extract profile-declared inline filters and the remaining question text."""
        names = (*profile.required_parameters, *profile.optional_parameters)
        filter_names = tuple(name for name in names if name != "question")
        if not filter_names:
            return ({"question": query}, (), ())

        pattern = re.compile(
            rf"\b(?P<name>{'|'.join(map(re.escape, filter_names))})\s*[:=]\s*"
            r"(?P<value>[^,;]+)",
            flags=re.IGNORECASE,
        )
        values: dict[str, list[str]] = {}
        for match in pattern.finditer(query):
            name = match.group("name").lower()
            value = match.group("value").strip()
            if value:
                values.setdefault(name, []).append(value)

        question = pattern.sub("", query).strip(" ,;")
        parameters = {name: entries[0] for name, entries in values.items()}
        if question:
            parameters["question"] = question
        ambiguity_flags = tuple(
            name
            for name, entries in values.items()
            if any(re.search(r"\s(?:or|and/or)\s|/", entry, flags=re.IGNORECASE) for entry in entries)
        )
        conflict_flags = tuple(
            name for name, entries in values.items() if len(set(entries)) > 1
        )
        return parameters, ambiguity_flags, conflict_flags

    @staticmethod
    def _validate(
        parameters: Mapping[str, str],
        profile: BenchmarkProfile,
        missing_parameters: tuple[str, ...],
    ) -> ValidationResult:
        """Validate present parameter values with declarative profile rules."""
        errors = [f"missing required parameter: {name}" for name in missing_parameters]
        for name, value in parameters.items():
            rule = profile.validation_rules.get(name)
            if rule and not re.fullmatch(rule, value):
                errors.append(f"invalid parameter: {name}")
        return ValidationResult(is_valid=not errors, errors=tuple(errors))

    @staticmethod
    def _confidence(
        missing_parameters: tuple[str, ...],
        ambiguity_flags: tuple[str, ...],
        conflict_flags: tuple[str, ...],
        validation_result: ValidationResult,
    ) -> float:
        """Calculate deterministic confidence from qualification evidence."""
        penalty = (
            0.35 * len(missing_parameters)
            + 0.20 * len(ambiguity_flags)
            + 0.25 * len(conflict_flags)
            + 0.10 * len(validation_result.errors)
        )
        return max(0.0, round(1.0 - penalty, 2))

    @staticmethod
    def _clarifications(
        missing_parameters: tuple[str, ...],
        ambiguity_flags: tuple[str, ...],
        conflict_flags: tuple[str, ...],
        confidence_score: float,
        profile: BenchmarkProfile,
    ) -> tuple[ClarificationRequest, ...]:
        """Return only specific structured clarification reasons allowed by the profile."""
        requests: list[ClarificationRequest] = []
        policy = profile.clarification_policy
        if policy.get("missing_required", False):
            requests.extend(ClarificationRequest(name, "missing_required") for name in missing_parameters)
        if policy.get("ambiguity", False):
            requests.extend(ClarificationRequest(name, "ambiguous") for name in ambiguity_flags)
        if policy.get("conflict", False):
            requests.extend(ClarificationRequest(name, "conflicting") for name in conflict_flags)
        if policy.get("low_confidence", False) and confidence_score < profile.confidence_threshold:
            requests.append(ClarificationRequest(None, "low_confidence"))
        return tuple(requests)
