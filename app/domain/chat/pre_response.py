
from email import message
from enum import Enum

from pydantic import BaseModel

class PreResponse(BaseModel):
    """Schema for pre-response situations"""
    message: str
    
    class Config:
        use_enum_values = True
    
    def __init__(self, **data):
        super().__init__(**data)

        
class PreResponsePresets:
    """Presets for common pre-response scenarios"""
    
    NO_RELEVANCE_PAPER = PreResponse(
        message="""I couldn't find any relevant research papers for your question. This could be because:

1. The topic might be too specific or recent
2. There may be no academic papers published on this subject
3. The papers may be behind paywalls

Please try asking a different question or rephrase your current one."""
    )
    
    GIBBERISH = PreResponse(
        message="""Hello! I'm exegent, an academic research assistant.

I'm here to help you explore and understand academic research papers! I can:

**Search** through millions of research papers across all disciplines  
**Analyze** and summarize complex scientific papers  
**Find** relevant citations and evidence for your questions  
**Compare** different research findings and methodologies  

**How to get started:**

Ask me clear research questions like:
- "What are the latest findings on climate change?"
- "How does machine learning improve medical diagnosis?"
- "What are the ethical implications of AI?"

**Tips for better results:**
- Be specific about what you want to know
- Use proper words and complete sentences
- Ask about scientific topics, research areas, or academic questions

Try asking me a research question, and I'll find and analyze relevant papers for you!"""
    )
    
    @classmethod
    def get_preset(cls, preset_name: str) -> PreResponse:
        presets = {
            "no_relevance_paper": PreResponsePresets.NO_RELEVANCE_PAPER,
            "gibberish": PreResponsePresets.GIBBERISH,
        }
        return presets.get(preset_name.lower(), cls.GIBBERISH)        
