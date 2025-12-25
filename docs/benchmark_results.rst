Benchmark Results
=================

.. raw:: html

   <div style="padding: 20px; background: #f8f9fa; border-radius: 8px; margin: 20px 0;">
       <h2 style="color: #667eea;">📊 Interactive Benchmark Report</h2>
       <p style="font-size: 1.1em;">
           View the complete benchmark results with interactive charts and detailed performance metrics.
       </p>
       <div style="margin-top: 20px;">
           <a href="_static/benchmark_report.html" 
              style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold;"
              onclick="window.open(this.href, '_blank', 'width=1200,height=800'); return false;">
              🚀 Open Full Report
           </a>
       </div>
       <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
           <strong>Note:</strong> The report opens in a new window. If it doesn't open automatically, 
           navigate to <code>docs/_build/html/_static/benchmark_report.html</code> and open it directly.
       </p>
   </div>

Quick Summary
-------------

The benchmark suite tests 30 different scenarios across 4 categories:

- **Insert Operations** (9 tests): Single and batch insert performance
- **Search Operations** (9 tests): Top-k search with varying k values  
- **Batch Operations** (6 tests): Combined operations and cycles
- **Scale Tests** (6 tests): Large-scale operations with 10K vectors

How to Generate Fresh Results
------------------------------

.. code-block:: bash

   # Run benchmarks (30-40 minutes)
   make benchmark
   
   # Generate HTML report
   make benchmark-report
   
   # Update documentation
   make benchmark-docs
   
   # Rebuild docs
   cd docs && make html

Vector Store Performance Highlights
------------------------------------

Based on the latest benchmark results:

**Qdrant**
   - Best for single inserts (~1ms)
   - Good batch performance
   - Moderate search speed

**Milvus**  
   - Slower for inserts (requires flush)
   - Fast search operations
   - Good for read-heavy workloads

**ChromaDB**
   - Balanced performance
   - In-memory, fast for small datasets
   - Good for development/testing

For detailed metrics, charts, and full analysis, view the interactive report above.

.. seealso::

   :doc:`benchmarks`
      Complete benchmark documentation with architecture details and usage guide.
