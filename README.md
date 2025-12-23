## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mushy07/AI-Agent-using-LangGraph.git
   cd AI-Agent-using-LangGraph
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Agent

### Start the interactive agent:

```bash
python core/simple_agent.py
```

Or from the root directory:

```bash
py .\core\simple_agent.py
```

### Using the Agent

1. The agent will start with a prompt: `You:`
2. Type your research question (e.g., "tell me about dogs", "what is the Eiffel Tower?")
3. The agent will:
   - Search the knowledge base
   - Display relevant content
   - Show source references
   - Display current state (conversation history, sources count, etc.)
4. Continue asking questions
5. Type `exit` to quit
