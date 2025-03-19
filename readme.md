# OpenAI Agents Project

[YouTube Tutorial Placeholder]

## Getting Started

### Prerequisites
- Python 3.11
- pipenv
- OpenAI API key
- OpenAI Vector Store ID
### Setup

1. Clone the repository:
```bash
git clone https://github.com/JacobD2001/openai-agents.git
cd openai-agents
```

2. Set up environment variables:
```bash
cp .env.example .env
```

3. Edit the `.env` file with your OpenAI API key and vector store id of your opean vector store as outlined in the video.

4. Install dependencies using pipenv:
```bash
pipenv install
```

5. Activate the pipenv environment:
```bash
pipenv shell
```

6. Run your agent application:
```bash
python main.py  # or the appropriate entry point for your application
```

## Documentation

For more information on how to use and implement agents, refer to the documentation: https://openai.github.io/openai-agents-python/agents/

1. [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs configured with instructions, tools, guardrails, and handoffs
2. [**Tools**](https://openai.github.io/openai-agents-python/tools/): Custom tools for agents to use
3. [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): Allow agents to transfer control to other agents for specific tasks
4. [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation
5. [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows