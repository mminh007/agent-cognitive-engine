#!/bin/sh

export PYTHONUNBUFFERED=1
# Set PYTHONPATH to the current working directory (/app) so Python can resolve 'app.xxxx' imports cleanly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. Run gRPC Server (Background process &)
# echo "🚀 Starting gRPC Server on port 50051..."
# python -m app.grpc_server &

# 2. Run FastAPI Web Server (Foreground process)
echo "🔥 Starting FastAPI Web Server on port 8000..."
exec "$@"