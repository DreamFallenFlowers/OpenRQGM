import asyncio
import json

from rqgm.toy import build_toy_engine, result_to_dict


async def main() -> None:
    engine = build_toy_engine()
    result = await engine.run()
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
