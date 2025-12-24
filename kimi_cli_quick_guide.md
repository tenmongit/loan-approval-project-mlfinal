# Kimi CLI Quick Guide

## What is Kimi CLI?
Kimi CLI is an interactive AI agent that runs on your computer and can help you with various tasks using natural language commands.

## Basic Usage

### 1. Natural Language Commands
Simply type what you want to do in plain English:
- "Read the README file"
- "List all Python files in the project"
- "Search for 'error' in log files"
- "Create a new Python script for data processing"

### 2. File Operations
```
Read files: "Read the file at src/main.py"
Write files: "Create a new file called config.json with basic settings"
Edit files: "Replace 'old_text' with 'new_text' in filename.py"
Search files: "Find all files containing 'import pandas'"
```

### 3. Code Execution
```
Run Python: "Execute this Python code: print('Hello World')"
Run shell: "List files in current directory"
Install packages: "Install numpy and pandas"
```

### 4. Web Operations
```
Search web: "Search for latest pandas documentation"
Fetch URLs: "Get content from https://example.com"
```

## Key Features

### Context Awareness
- Kimi remembers your conversation history
- Works within your current project directory
- Understands file structures and project context

### Tool Integration
- **Shell**: Run system commands
- **File operations**: Read, write, edit files
- **Code execution**: Python and shell scripts
- **Web search**: Get latest information
- **Git operations**: Repository management

### Safety Features
- Works within your project directory by default
- Asks for confirmation before major changes
- Provides clear feedback on actions taken

## Common Patterns

### 1. Project Exploration
```
"What files are in this project?"
"Show me the project structure"
"Read the main README file"
```

### 2. Code Development
```
"Create a Python script to process CSV files"
"Fix this error in my code: [paste error]"
"Add error handling to this function"
```

### 3. Data Analysis
```
"Analyze this CSV file and show summary statistics"
"Create visualizations for this data"
"Clean and preprocess this dataset"
```

### 4. Debugging
```
"Help me debug this Python script"
"What's causing this import error?"
"Check if all required packages are installed"
```

## Best Practices

1. **Be Specific**: Instead of "fix this", say "fix the syntax error on line 15"
2. **Provide Context**: Include file paths, error messages, expected behavior
3. **Ask for Confirmation**: Kimi will ask before making major changes
4. **Check Results**: Review the output and ask for clarification if needed

## Example Workflow

```
User: "What files are in this project?"
Kimi: Lists directory contents

User: "Read the README file"
Kimi: Shows README content

User: "Create a Python script to analyze the data"
Kimi: Creates analysis script

User: "Run the script and show results"
Kimi: Executes script and displays output
```

## Getting Help

If you're unsure how to do something, just ask:
- "How do I search for files?"
- "What's the best way to edit this file?"
- "Can you help me understand this error?"

Remember: Kimi CLI is designed to understand natural language, so you don't need to learn special commands - just describe what you want to accomplish!