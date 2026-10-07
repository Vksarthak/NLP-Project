import logging
import time
from typing import Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class RateLimitError(LLMError):
    """Custom exception for rate limit errors."""
    pass

class GeminiClient:
    """Client for interacting with the Gemini API."""

    def __init__(self, api_key: str, model: str = 'gemini-3.5-flash-lite'):
        """
        Initialize the GeminiClient.

        Args:
            api_key: The Google API key.
            model: The Gemini model to use.
        """
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.total_calls = 0

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """
        Generate content using the Gemini API.

        Args:
            prompt: The input prompt.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The generated response string.

        Raises:
            RateLimitError: If rate limited after retries.
            LLMError: For other API errors.
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                self.total_calls += 1
                logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries}). Total calls: {self.total_calls}")
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                
                logger.info("Received response from Gemini API")
                
                if response.text is None:
                    raise LLMError("Received empty response from API")
                    
                return response.text
                
            except APIError as e:
                # HTTP 429 indicates Too Many Requests (Rate Limit)
                if e.code == 429:
                    logger.warning(f"Rate limited by API. Attempt {attempt + 1} of {max_retries}")
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"Rate limit exceeded after {max_retries} retries: {e}")
                    time.sleep(base_delay * (2 ** attempt))
                # Transient server errors
                elif e.code in [500, 502, 503, 504]:
                    logger.warning(f"Transient error: {e}. Attempt {attempt + 1} of {max_retries}")
                    if attempt == max_retries - 1:
                        raise LLMError(f"Transient error persisted after {max_retries} retries: {e}")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    logger.error(f"API Error: {e}")
                    raise LLMError(f"API Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise LLMError(f"Unexpected error: {e}")
