#!/bin/bash

# Configuration
BACKEND_DIR="/Users/alexandergruhl/Development/GenerativeAgents"
FRONTEND_DIR="/Users/alexandergruhl/Development/GenerativeAgents/frontend"

MODE=$1

# Function to kill background processes on exit
cleanup() {
    echo "Stopping sandbox..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT

echo "Starting Generative Agents Sandbox..."

# Start Backend
echo "Starting Backend in $BACKEND_DIR..."
cd "$BACKEND_DIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Check for venv
if [ -d ".venv" ]; then
    PYTHON_EXEC=".venv/bin/python"
    echo "Using virtual environment: .venv"
else
    PYTHON_EXEC="python3"
fi

if [ "$MODE" == "mock" ]; then
    echo "Running in **MOCK** mode..."
    $PYTHON_EXEC -m generative_agents.mock_server &
else
    echo "Running in **REAL** mode..."
    $PYTHON_EXEC -m generative_agents &
fi
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start Frontend
echo "Starting Frontend in $FRONTEND_DIR..."
cd "$FRONTEND_DIR"
# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start Vite dev server
npm run dev &
FRONTEND_PID=$!

echo "Sandbox running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop."

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
