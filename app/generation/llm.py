"""LLM interface supporting Groq, OpenAI, and HuggingFace providers."""

_GROQ_MODELS = {
    "openai/gpt-oss-120b",
}
"""Known supported Groq model names. Used for reference and validation."""


def get_llm(
    provider: str = "groq",
    model_name: str = "openai/gpt-oss-120b",
    **kwargs,
):
    """Return a LangChain LLM object for the specified provider.

    Args:
        provider: LLM provider, one of "groq", "openai", or "huggingface".
        model_name: Name of the model to use.
        **kwargs: Additional keyword arguments (e.g., temperature, max_tokens,
            groq_api_key, openai_api_key).

    Returns:
        LangChain LLM object (ChatGroq, ChatOpenAI, or HuggingFacePipeline).

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    temperature = kwargs.get("temperature", 0.0)
    max_tokens = kwargs.get("max_tokens", 512)

    if provider == "groq":
        from langchain_groq import ChatGroq

        if model_name not in _GROQ_MODELS:
            raise ValueError(
                f"Unsupported Groq model: {model_name}. "
                f"Supported models: {sorted(_GROQ_MODELS)}"
            )
        groq_api_key = kwargs.get("groq_api_key")
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "huggingface":
        from transformers import pipeline as hf_pipeline
        from langchain_community.llms import HuggingFacePipeline

        pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Use 'groq', 'openai', or 'huggingface'."
        )
