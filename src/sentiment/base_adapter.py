from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSentimentAdapter(ABC):
    """
    Abstract Base Class for all financial sentiment ingestion adapters.
    Ensures all adapters parse their sources into the unified opinion schema.
    """
    
    @abstractmethod
    def fetch_recent_data(self) -> Any:
        """Fetch the raw raw data from the external source (e.g. RSS, PDF, Twitter API)."""
        pass
        
    @abstractmethod
    def extract_opinions(self, raw_data: Any, model_name: str = "gpt-4o") -> List[Dict[str, Any]]:
        """
        Use an LLM to analyze raw content and extract structured ticker opinions
        matching the Unified Analyst Opinion Schema.
        """
        pass
