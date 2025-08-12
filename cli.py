import asyncio
import json
import sys
import argparse
from cmr_agent.graph.pipeline import build_graph

async def main():
    parser = argparse.ArgumentParser(description='NASA CMR AI Agent CLI')
    parser.add_argument('query', nargs='*', help='User query')
    parser.add_argument('--stream', action='store_true', help='Stream step-wise updates')
    parser.add_argument('--json', action='store_true', help='Print minimal JSON summary')
    args = parser.parse_args()

    query = ' '.join(args.query) or 'Find precipitation datasets for Sub-Saharan Africa 2015-2023'
    graph = build_graph()

    if args.stream:
        # Stream AST events
        async for event in graph.astream({'user_query': query}):
            print(event)
        return

    result = await graph.ainvoke({'user_query': query})
    if args.json:
        # minimal summary
        summary = {
            'intent': result.get('intent'),
            'total_collections': (result.get('analysis') or {}).get('total_collections'),
            'total_granules': (result.get('analysis') or {}).get('total_granules'),
            'bbox': ((result.get('analysis') or {}).get('queries') or [{}])[0].get('spatial_extent'),
            'coverage': ((result.get('analysis') or {}).get('queries') or [{}])[0].get('temporal_coverage'),
        }
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
