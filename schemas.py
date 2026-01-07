from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):

    """This will be defining the Source URL of the Response"""

    url : str = Field(description="The URL of the Source")

class AgentResponse(BaseModel):

    """This will be containing the Answer and the List of Source URL's"""

    answer : str = Field(description="The Answer give by the Agent")

    sources : List[Source] = Field(default_factory=list, description="List of Source URL")