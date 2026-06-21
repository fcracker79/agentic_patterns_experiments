import asyncio
from fastmcp import Client


async def _main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        print(await client.list_tools())


if __name__ == "__main__":
    asyncio.run(_main())
