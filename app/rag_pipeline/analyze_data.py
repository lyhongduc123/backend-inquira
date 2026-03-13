"""
RAG Pipeline Data Analyzer

Utility script to view, analyze, and compare collected pipeline execution data.

Usage:
    python -m app.rag_pipeline.analyze_data --latest
    python -m app.rag_pipeline.analyze_data --file database_20260306_143022_123456.json
    python -m app.rag_pipeline.analyze_data --compare file1.json file2.json
    python -m app.rag_pipeline.analyze_data --stats
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class DataAnalyzer:
    """Analyze and visualize RAG pipeline execution data."""
    
    def __init__(self, logs_dir: str = "app/rag_pipeline/logs"):
        self.logs_dir = Path(logs_dir)
    
    def get_latest_file(self) -> Path:
        """Get the most recent execution data file."""
        files = sorted(self.logs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No execution data files found in {self.logs_dir}")
        return files[0]
    
    def load_execution(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load execution data from file."""
        file_path: Path
        
        if filepath is None:
            file_path = self.get_latest_file()
        else:
            file_path = self.logs_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a formatted summary of execution data."""
        print("\n" + "="*80)
        print(f"RAG Pipeline Execution Summary")
        print("="*80)
        
        print(f"\n📋 Metadata:")
        print(f"  Execution ID: {data['execution_id']}")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Pipeline Type: {data['pipeline_type']}")
        print(f"  Total Time: {data.get('total_time_ms', 0):.2f}ms")
        
        print(f"\n❓ Query Information:")
        print(f"  Original Query: {data['original_query']}")
        if data.get('intent'):
            print(f"  Detected Intent: {data['intent']}")
        if data.get('decomposed_queries'):
            print(f"  Decomposed Queries ({len(data['decomposed_queries'])}):")
            for i, q in enumerate(data['decomposed_queries'], 1):
                print(f"    {i}. {q}")
        
        if data.get('filters'):
            print(f"\n🔍 Filters Applied:")
            for key, value in data['filters'].items():
                if value is not None:
                    print(f"    {key}: {value}")
        
        print(f"\n📄 Retrieved Papers: {len(data.get('retrieved_papers', []))}")
        if data.get('retrieved_papers'):
            print(f"  Top 5 Papers by Search Score:")
            for i, paper in enumerate(data['retrieved_papers'][:5], 1):
                score = paper.get('search_score', 0)
                print(f"    {i}. [{paper['year']}] {paper['title'][:70]}...")
                print(f"       Score: {score:.4f}, Citations: {paper['citation_count']}")
        
        print(f"\n📝 Retrieved Chunks: {len(data.get('retrieved_chunks', []))}")
        if data.get('retrieved_chunks'):
            print(f"  Top 5 Chunks by Relevance:")
            for i, chunk in enumerate(data['retrieved_chunks'][:5], 1):
                score = chunk.get('relevance_score', 0)
                print(f"    {i}. {chunk['text'][:80]}...")
                print(f"       Paper: {chunk['paper_id']}, Relevance: {score:.4f}")
        
        print(f"\n🏆 Ranked Papers: {len(data.get('ranked_papers', []))}")
        if data.get('ranked_papers'):
            print(f"  Top 10 Papers by Final Ranking:")
            for i, paper in enumerate(data['ranked_papers'][:10], 1):
                scores = paper.get('ranking_scores', {})
                print(f"    {i}. [{paper.get('year')}] {paper.get('title', 'N/A')[:60]}...")
                print(f"       Final Score: {paper.get('relevance_score', 0):.4f}")
                if scores:
                    score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
                    print(f"       Component Scores: {score_str}")
        
        if data.get('ranking_weights'):
            print(f"\n⚖️  Ranking Weights:")
            for key, value in data['ranking_weights'].items():
                print(f"    {key}: {value}")
        
        print(f"\n⏱️  Timing Breakdown:")
        if data.get('decomposition_time_ms'):
            print(f"  Decomposition: {data['decomposition_time_ms']:.2f}ms")
        if data.get('search_time_ms'):
            print(f"  Search: {data['search_time_ms']:.2f}ms")
        if data.get('ranking_time_ms'):
            print(f"  Ranking: {data['ranking_time_ms']:.2f}ms")
        print(f"  Total: {data.get('total_time_ms', 0):.2f}ms")
        
        if data.get('errors'):
            print(f"\n❌ Errors ({len(data['errors'])}):")
            for error in data['errors']:
                print(f"  - {error}")
        
        if data.get('breakdown_response'):
            print(f"\n🔬 Breakdown Response:")
            br = data['breakdown_response']
            if br.get('intent'):
                print(f"  Intent: {br['intent']}")
            if br.get('search_queries'):
                print(f"  Search Queries: {br['search_queries']}")
            if br.get('filters'):
                print(f"  Extracted Filters: {br['filters']}")
            if br.get('reasoning'):
                print(f"  Reasoning: {br['reasoning']}")
        
        print("\n" + "="*80 + "\n")
    
    def print_statistics(self):
        """Print statistics across all executions."""
        files = list(self.logs_dir.glob("*.json"))
        if not files:
            print(f"No execution data files found in {self.logs_dir}")
            return
        
        print("\n" + "="*80)
        print(f"RAG Pipeline Statistics ({len(files)} executions)")
        print("="*80)
        
        total_times = []
        intent_counts = defaultdict(int)
        pipeline_counts = defaultdict(int)
        paper_counts = []
        chunk_counts = []
        
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if data.get('total_time_ms'):
                    total_times.append(data['total_time_ms'])
                
                if data.get('intent'):
                    intent_counts[data['intent']] += 1
                
                pipeline_counts[data['pipeline_type']] += 1
                
                paper_counts.append(len(data.get('retrieved_papers', [])))
                chunk_counts.append(len(data.get('retrieved_chunks', [])))
        
        print(f"\n📊 Execution Statistics:")
        print(f"  Total Executions: {len(files)}")
        print(f"  Average Time: {sum(total_times)/len(total_times):.2f}ms" if total_times else "  Average Time: N/A")
        print(f"  Min Time: {min(total_times):.2f}ms" if total_times else "  Min Time: N/A")
        print(f"  Max Time: {max(total_times):.2f}ms" if total_times else "  Max Time: N/A")
        
        print(f"\n🔧 Pipeline Types:")
        for pipeline, count in pipeline_counts.items():
            print(f"  {pipeline}: {count} ({count/len(files)*100:.1f}%)")
        
        print(f"\n🎯 Intent Distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent}: {count} ({count/len(files)*100:.1f}%)")
        
        print(f"\n📄 Paper Statistics:")
        print(f"  Average Papers Retrieved: {sum(paper_counts)/len(paper_counts):.1f}" if paper_counts else "  Average Papers: N/A")
        print(f"  Min Papers: {min(paper_counts)}" if paper_counts else "  Min Papers: N/A")
        print(f"  Max Papers: {max(paper_counts)}" if paper_counts else "  Max Papers: N/A")
        
        print(f"\n📝 Chunk Statistics:")
        print(f"  Average Chunks Retrieved: {sum(chunk_counts)/len(chunk_counts):.1f}" if chunk_counts else "  Average Chunks: N/A")
        print(f"  Min Chunks: {min(chunk_counts)}" if chunk_counts else "  Min Chunks: N/A")
        print(f"  Max Chunks: {max(chunk_counts)}" if chunk_counts else "  Max Chunks: N/A")
        
        print("\n" + "="*80 + "\n")
    
    def compare_executions(self, file1: str, file2: str):
        """Compare two execution data files."""
        data1 = self.load_execution(file1)
        data2 = self.load_execution(file2)
        
        print("\n" + "="*80)
        print(f"Comparing Pipeline Executions")
        print("="*80)
        
        print(f"\nExecution 1: {data1['execution_id']}")
        print(f"  Query: {data1['original_query']}")
        print(f"  Time: {data1.get('total_time_ms', 0):.2f}ms")
        print(f"  Papers: {len(data1.get('retrieved_papers', []))}")
        print(f"  Chunks: {len(data1.get('retrieved_chunks', []))}")
        
        print(f"\nExecution 2: {data2['execution_id']}")
        print(f"  Query: {data2['original_query']}")
        print(f"  Time: {data2.get('total_time_ms', 0):.2f}ms")
        print(f"  Papers: {len(data2.get('retrieved_papers', []))}")
        print(f"  Chunks: {len(data2.get('retrieved_chunks', []))}")
        
        print(f"\n📊 Comparison:")
        time_diff = data2.get('total_time_ms', 0) - data1.get('total_time_ms', 0)
        paper_diff = len(data2.get('retrieved_papers', [])) - len(data1.get('retrieved_papers', []))
        chunk_diff = len(data2.get('retrieved_chunks', [])) - len(data1.get('retrieved_chunks', []))
        
        print(f"  Time Difference: {time_diff:+.2f}ms ({time_diff/data1.get('total_time_ms', 1)*100:+.1f}%)")
        print(f"  Paper Difference: {paper_diff:+d}")
        print(f"  Chunk Difference: {chunk_diff:+d}")
        
        # Compare top papers
        papers1_ids = {p['paper_id'] for p in data1.get('ranked_papers', [])[:10]}
        papers2_ids = {p['paper_id'] for p in data2.get('ranked_papers', [])[:10]}
        
        overlap = papers1_ids & papers2_ids
        print(f"\n📄 Top 10 Papers Overlap: {len(overlap)}/10")
        
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG pipeline execution data")
    parser.add_argument('--latest', action='store_true', help='Show the latest execution')
    parser.add_argument('--file', type=str, help='Show specific execution file')
    parser.add_argument('--stats', action='store_true', help='Show statistics across all executions')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'), help='Compare two executions')
    parser.add_argument('--logs-dir', type=str, default='app/rag_pipeline/logs', help='Directory containing log files')
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(logs_dir=args.logs_dir)
    
    try:
        if args.stats:
            analyzer.print_statistics()
        elif args.compare:
            analyzer.compare_executions(args.compare[0], args.compare[1])
        elif args.file:
            data = analyzer.load_execution(args.file)
            analyzer.print_summary(data)
        elif args.latest:
            data = analyzer.load_execution()
            analyzer.print_summary(data)
        else:
            # Default: show latest
            data = analyzer.load_execution()
            analyzer.print_summary(data)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error analyzing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
