import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List
from ..base import BaseSynthesizer
from ..prompts import SYSTEM_PROMPT

class Gemma31BSynthesizer(BaseSynthesizer):
    def __init__(self, is_batch_mode: bool = False):
        """
        Initializes the Gemma 4 31B model using the google-genai SDK.
        """
        load_dotenv()
        self.api_key = os.getenv("API_KEY_GEMINI")
        
        if not self.api_key:
            raise ValueError("API_KEY_GEMINI not found in .env file.")
        
        # Initialize the new Client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemma-4-31b-it"
        self.is_batch_mode = is_batch_mode

    def synthesize(self, texts: List[str]) -> str:
        if not texts:
            return "Insufficient data for synthesis."
            
        # Format the inputs cleanly
        input_data = "Here are the VLM outputs to synthesize:\n\n"
        for i, text in enumerate(texts):
            input_data += f"--- OUTPUT {i+1} ---\n{text}\n\n"

        # Retry logic for API errors

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
        
            try:
                # Call the API client with the new syntax
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=input_data,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        top_p=0.1
                    )
                )
                
                # Apply pacing for the initial database generation
                if self.is_batch_mode:
                    time.sleep(4.1) 
                    
                return response.text
                
            except Exception as e:

                if attempt == max_retries - 1:
                    print(f"\nDefinitive synthesis error after {max_retries} attempts: {str(e)}")
                    raise e
                
                wait_time = base_delay * (2 ** attempt)
                print(f"\nServer Error. Pausing {wait_time}s before retryying... (Try {attempt + 1}/{max_retries})")
                time.sleep(wait_time)