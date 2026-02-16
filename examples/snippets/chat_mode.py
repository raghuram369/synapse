from synapse import Synapse
s = Synapse('chat-bot')
# In your message handler:
if s.command_parser.is_memory_command(user_message):
    response = s.command(user_message)  # handles /mem remember, /mem recall, etc.
