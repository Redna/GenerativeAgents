import asyncio
import socketio

async def verify_emission():
    sio = socketio.AsyncClient()
    
    @sio.on('connect')
    async def on_connect():
        print("Connected to server")
        print("Subscribing to Klaus_Mueller...")
        await sio.emit('subscribe_agent_log', {'agent_name': 'Klaus_Mueller'})

    @sio.on('agent_log')
    async def on_agent_log(data):
        print(f"RECEIVED LOG: {data}")

    @sio.on('update')
    async def on_update(data):
        # Just to see if updates are coming (heartbeat)
        sys.stdout.write(".")
        sys.stdout.flush()

    import sys
    print("Connecting...")
    try:
        await sio.connect('http://localhost:8000')
        print("Waiting for logs (Ctrl+C to stop)...")
        await asyncio.sleep(20) # Listen for 20 seconds
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await sio.disconnect()

if __name__ == "__main__":
    asyncio.run(verify_emission())
