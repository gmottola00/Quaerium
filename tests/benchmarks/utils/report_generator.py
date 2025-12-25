"""Generate HTML reports from benchmark results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BenchmarkReportGenerator:
    """Generate HTML reports from pytest-benchmark JSON output."""
    
    def __init__(self, json_path: str | Path):
        """Initialize report generator.
        
        Args:
            json_path: Path to pytest-benchmark JSON file
        """
        self.json_path = Path(json_path)
        with open(self.json_path) as f:
            self.data = json.load(f)
    
    def generate_html(self, output_path: str | Path) -> None:
        """Generate HTML report.
        
        Args:
            output_path: Path to save HTML report
        """
        output_path = Path(output_path)
        html = self._build_html()
        output_path.write_text(html)
    
    def _build_html(self) -> str:
        """Build complete HTML report."""
        benchmarks = self.data.get("benchmarks", [])
        
        # Group benchmarks by operation type
        grouped = self._group_benchmarks(benchmarks)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Store Performance Benchmarks</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 2rem;
        }}
        
        h1 {{
            color: #2d3748;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }}
        
        .subtitle {{
            color: #718096;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            opacity: 0.9;
        }}
        
        .summary-card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        
        .section {{
            margin-bottom: 3rem;
        }}
        
        .section h2 {{
            color: #2d3748;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 2rem;
            padding: 1rem;
            background: #f7fafc;
            border-radius: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            color: #4a5568;
        }}
        
        tr:hover {{
            background: #f7fafc;
        }}
        
        .metric {{
            font-weight: 600;
            color: #2d3748;
        }}
        
        .vector-store {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        
        .milvus {{
            background: #e6fffa;
            color: #047857;
        }}
        
        .qdrant {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .chroma {{
            background: #dbeafe;
            color: #1e40af;
        }}
        
        .footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Vector Store Performance Benchmarks</h1>
        <p class="subtitle">Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
        
        {self._generate_summary(benchmarks)}
        
        {self._generate_sections(grouped)}
        
        <div class="footer">
            <p>Generated by RAG Toolkit Benchmark Suite</p>
            <p>Powered by pytest-benchmark</p>
        </div>
    </div>
    
    {self._generate_scripts(grouped)}
</body>
</html>"""
        return html
    
    def _group_benchmarks(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group benchmarks by operation type."""
        grouped = {}
        
        for benchmark in benchmarks:
            name = benchmark.get("name", "")
            
            # Determine group based on name
            if "insert" in name.lower():
                group = "Insert Operations"
            elif "search" in name.lower():
                group = "Search Operations"
            elif "batch" in name.lower():
                group = "Batch Operations"
            elif "scale" in name.lower():
                group = "Scalability Tests"
            else:
                group = "Other"
            
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(benchmark)
        
        return grouped
    
    def _generate_summary(self, benchmarks: List[Dict[str, Any]]) -> str:
        """Generate summary cards."""
        total_benchmarks = len(benchmarks)
        
        # Calculate statistics
        all_means = [b["stats"]["mean"] for b in benchmarks]
        fastest = min(all_means) if all_means else 0
        slowest = max(all_means) if all_means else 0
        avg_time = sum(all_means) / len(all_means) if all_means else 0
        
        return f"""
        <div class="summary">
            <div class="summary-card">
                <h3>Total Benchmarks</h3>
                <div class="value">{total_benchmarks}</div>
            </div>
            <div class="summary-card">
                <h3>Fastest Operation</h3>
                <div class="value">{fastest*1000:.2f} ms</div>
            </div>
            <div class="summary-card">
                <h3>Slowest Operation</h3>
                <div class="value">{slowest*1000:.2f} ms</div>
            </div>
            <div class="summary-card">
                <h3>Average Time</h3>
                <div class="value">{avg_time*1000:.2f} ms</div>
            </div>
        </div>
        """
    
    def _generate_sections(self, grouped: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate sections for each benchmark group."""
        sections_html = ""
        
        for idx, (group, benchmarks) in enumerate(grouped.items()):
            chart_id = f"chart{idx}"
            
            sections_html += f"""
            <div class="section">
                <h2>{group}</h2>
                <div class="chart-container">
                    <canvas id="{chart_id}"></canvas>
                </div>
                {self._generate_table(benchmarks)}
            </div>
            """
        
        return sections_html
    
    def _generate_table(self, benchmarks: List[Dict[str, Any]]) -> str:
        """Generate detailed table for benchmarks."""
        rows = ""
        
        for benchmark in benchmarks:
            name = benchmark.get("name", "")
            stats = benchmark.get("stats", {})
            
            # Extract vector store name
            store = "unknown"
            if "milvus" in name.lower():
                store = "milvus"
            elif "qdrant" in name.lower():
                store = "qdrant"
            elif "chroma" in name.lower():
                store = "chroma"
            
            rows += f"""
            <tr>
                <td><span class="vector-store {store}">{store.upper()}</span></td>
                <td>{name}</td>
                <td class="metric">{stats.get('mean', 0)*1000:.2f} ms</td>
                <td>{stats.get('stddev', 0)*1000:.2f} ms</td>
                <td>{stats.get('min', 0)*1000:.2f} ms</td>
                <td>{stats.get('max', 0)*1000:.2f} ms</td>
                <td class="metric">{1/stats.get('mean', 1):.2f} ops/sec</td>
            </tr>
            """
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Store</th>
                    <th>Benchmark</th>
                    <th>Mean</th>
                    <th>StdDev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Ops/Sec</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
    
    def _generate_scripts(self, grouped: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate Chart.js scripts."""
        scripts = "<script>"
        
        for idx, (group, benchmarks) in enumerate(grouped.items()):
            chart_id = f"chart{idx}"
            
            # Prepare data for chart
            labels = [b.get("name", "") for b in benchmarks]
            means = [b["stats"]["mean"] * 1000 for b in benchmarks]  # Convert to ms
            
            # Assign colors based on vector store
            colors = []
            for label in labels:
                if "milvus" in label.lower():
                    colors.append("rgba(16, 185, 129, 0.8)")
                elif "qdrant" in label.lower():
                    colors.append("rgba(251, 191, 36, 0.8)")
                elif "chroma" in label.lower():
                    colors.append("rgba(59, 130, 246, 0.8)")
                else:
                    colors.append("rgba(156, 163, 175, 0.8)")
            
            scripts += f"""
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: 'Mean Time (ms)',
                        data: {json.dumps(means)},
                        backgroundColor: {json.dumps(colors)},
                        borderColor: {json.dumps(colors)},
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        title: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Time (ms)'
                            }}
                        }}
                    }}
                }}
            }});
            """
        
        scripts += "</script>"
        return scripts


__all__ = ["BenchmarkReportGenerator"]
