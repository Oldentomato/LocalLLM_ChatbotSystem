import nest_asyncio
from pyngrok import ngrok
import uvicorn

def set_external():
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()

if __name__ == '__main__': 
    # set_external()
    uvicorn.run(app="api.api:app" ,host="0.0.0.0", reload=True)