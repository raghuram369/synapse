#!/usr/bin/env python3
"""Demo script that simulates typing for terminal recording."""
import sys
import time

def type_out(text, delay=0.04):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')
    sys.stdout.flush()

def pause(seconds=1.0):
    time.sleep(seconds)

# Simulate a terminal session
print("\033[1;32m$\033[0m ", end="")
type_out("pip install synapse-memory")
pause(0.5)
print("Successfully installed synapse-memory-0.1.0")
pause(1.0)

print()
print("\033[1;32m$\033[0m ", end="")
type_out("python3")
pause(0.5)
print("Python 3.14.3 (main) [Clang 17.0.0]")
print('Type "help" for more information.')

print("\033[1;33m>>>\033[0m ", end="")
type_out("from synapse import Synapse")
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('s = Synapse()')
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('s.remember("I prefer vegetarian food and avoid dairy")')
pause(0.3)
print("Memory(id=1, content='I prefer vegetarian food and avoid dairy'...)")
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('s.remember("Meeting with Sarah at 3pm tomorrow about the project")')
pause(0.3)
print("Memory(id=2, content='Meeting with Sarah at 3pm tomorrow about the project'...)")
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('s.remember("Started learning piano last month, practicing daily")')
pause(0.3)
print("Memory(id=3, content='Started learning piano last month, practicing daily'...)")
pause(0.5)

print()
print("\033[1;33m>>>\033[0m ", end="")
type_out('# Now recall semantically — no API calls, pure local')
pause(0.5)

print("\033[1;33m>>>\033[0m ", end="")
type_out('results = s.recall("What are my dietary preferences?")')
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('print(results[0].content)')
pause(0.3)
print("\033[1;36mI prefer vegetarian food and avoid dairy\033[0m")
pause(0.8)

print()
print("\033[1;33m>>>\033[0m ", end="")
type_out('results = s.recall("What hobbies am I learning?")')
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('print(results[0].content)')
pause(0.3)
print("\033[1;36mStarted learning piano last month, practicing daily\033[0m")
pause(0.8)

print()
print("\033[1;33m>>>\033[0m ", end="")
type_out('results = s.recall("upcoming meetings")')
pause(0.3)

print("\033[1;33m>>>\033[0m ", end="")
type_out('print(results[0].content)')
pause(0.3)
print("\033[1;36mMeeting with Sarah at 3pm tomorrow about the project\033[0m")
pause(1.0)

print()
print("\033[1;33m>>>\033[0m ", end="")
type_out("# Zero API calls. Zero dependencies. Sub-millisecond recall. 🧠")
pause(2.0)
