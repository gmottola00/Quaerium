#!/usr/bin/env python3
"""Generate HTML report from benchmark results."""

import glob
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from benchmarks.utils.report_generator import BenchmarkReportGenerator


def main():
    """Generate benchmark report."""
    # Find most recent benchmark results
    benchmark_files = sorted(glob.glob(".benchmarks/*/0*.json"))
    
    if not benchmark_files:
        print("❌ No benchmark data found.")
        print("   Run 'make benchmark' first to generate benchmark data.")
        sys.exit(1)
    
    # Use most recent file
    latest_file = benchmark_files[-1]
    print(f"📊 Using benchmark data: {latest_file}")
    
    # Generate report
    output_file = "benchmark_report.html"
    try:
        gen = BenchmarkReportGenerator(latest_file)
        gen.generate_html(output_file)
        print(f"✅ Report generated: {output_file}")
        
        # Try to open the report
        if sys.platform == "darwin":  # macOS
            os.system(f"open {output_file}")
        elif sys.platform.startswith("linux"):  # Linux
            os.system(f"xdg-open {output_file}")
        else:
            print(f"   Open {output_file} in your browser")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
