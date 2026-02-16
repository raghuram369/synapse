from synapse import Synapse
from integrations.langchain import SynapseMemory
memory = SynapseMemory(synapse=Synapse('langchain-agent'))
# Drop into any LangChain chain as memory=memory
chain.memory = memory
