"""Domain failures emitted by the production indexing runtime."""


class IndexPipelineError(RuntimeError):
    """Base error for a failed indexing operation."""


class DocumentRoleAuthorizationError(IndexPipelineError):
    """Raised before indexing when a source is not reference knowledge."""


class CollectionAlreadyExists(IndexPipelineError):
    """Raised when a caller requests creation of an existing collection."""


class CollectionConfigurationMismatch(IndexPipelineError):
    """Raised when an existing collection is incompatible with this indexer."""


class EmbeddingGenerationError(IndexPipelineError):
    """Raised when the configured embedding provider cannot produce vectors."""


class VectorWriteError(IndexPipelineError):
    """Raised when a batch cannot be persisted after retrying."""
