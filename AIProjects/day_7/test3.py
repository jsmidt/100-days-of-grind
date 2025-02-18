import asyncio

async def fetch_data():
    print("Fetching data...")
    await asyncio.sleep(3)  # Non-blocking
    print("Data fetched!")

async def do_something_else():
    for i in range(3):
        await asyncio.sleep(1)  # Non-blocking
        print(f"Doing something else... {i+1}")

async def main():
    await asyncio.gather(fetch_data(), do_something_else())  # Run both concurrently
    print("Done!")

asyncio.run(main())
