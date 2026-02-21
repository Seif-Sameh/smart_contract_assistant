"""LLM interface supporting OpenAI and HuggingFace providers."""


def get_llm(
    provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    **kwargs,
):
    """Return a LangChain LLM object for the specified provider.

    Args:
        provider: LLM provider, one of "openai" or "huggingface".
        model_name: Name of the model to use.
        **kwargs: Additional keyword arguments (e.g., temperature, max_tokens).

    Returns:
        LangChain LLM object (ChatOpenAI or HuggingFacePipeline).

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    temperature = kwargs.get("temperature", 0.0)
    max_tokens = kwargs.get("max_tokens", 512)

    if provider == "openai":
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
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'huggingface'.")
