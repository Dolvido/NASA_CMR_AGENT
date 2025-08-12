import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.pop('OPENAI_API_KEY', None)
os.environ.pop('ANTHROPIC_API_KEY', None)
