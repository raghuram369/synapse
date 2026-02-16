from synapse import Synapse
s = Synapse('my-agent')
s.remember('User prefers dark mode')
results = s.recall('what theme?')
context = s.compile_context('user preferences', budget=2000)
