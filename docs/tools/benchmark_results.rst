Benchmark Results
=================

.. raw:: html

   <style>
       .benchmark-hero {
           background: #667eea;
           padding: 40px;
           border-radius: 8px;
           margin: 30px 0;
           text-align: center;
       }
       .benchmark-hero h2 {
           color: white;
           font-size: 2em;
           margin: 0 0 15px 0;
           font-weight: 600;
       }
       .benchmark-hero p {
           color: rgba(255, 255, 255, 0.9);
           font-size: 1.1em;
           margin: 0 0 25px 0;
       }
       .benchmark-button {
           display: inline-block;
           padding: 12px 32px;
           background: white;
           color: #667eea;
           text-decoration: none;
           border-radius: 6px;
           font-weight: 600;
           transition: opacity 0.2s;
       }
       .benchmark-button:hover {
           opacity: 0.9;
       }
       
       /* Dark mode support */
       [data-theme="dark"] .benchmark-button {
           background: #2b2b2b;
           color: #8b9cf6;
       }
       
       .benchmark-stats {
           display: grid;
           grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 15px;
           margin: 30px 0;
       }
       .stat-card {
           background: var(--color-background-secondary, #f8f9fa);
           padding: 20px;
           border-radius: 6px;
           text-align: center;
           border: 1px solid var(--color-background-border, #e0e0e0);
       }
       .stat-number {
           font-size: 2.5em;
           font-weight: 600;
           color: #667eea;
           margin: 0;
       }
       .stat-label {
           font-size: 0.95em;
           color: var(--color-foreground-secondary, #666);
           margin-top: 8px;
       }
       .info-note {
           background: var(--color-admonition-background, rgba(102, 126, 234, 0.1));
           border-left: 3px solid #667eea;
           padding: 15px 20px;
           border-radius: 4px;
           margin: 20px 0;
       }
       .info-note p {
           color: var(--color-foreground-primary, inherit);
       }
       .section-simple {
           margin: 30px 0;
       }
       .simple-grid {
           display: grid;
           grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
           gap: 20px;
           margin: 20px 0;
       }
       .simple-card {
           background: var(--color-background-secondary, #fafafa);
           border: 1px solid var(--color-background-border, #e0e0e0);
           padding: 20px;
           border-radius: 6px;
       }
       .simple-card h4 {
           margin: 0 0 12px 0;
           color: var(--color-foreground-primary, #333);
           font-size: 1.2em;
       }
       .simple-card ul {
           margin: 0;
           padding-left: 20px;
           color: var(--color-foreground-secondary, #666);
           line-height: 1.8;
       }
   </style>

   <div class="benchmark-hero">
       <h2>📊 Interactive Benchmark Report</h2>
       <p>
           Performance comparison of Milvus, Qdrant, and ChromaDB across 30 tests
       </p>
       <a href="_static/benchmark_report.html" 
          class="benchmark-button"
          onclick="window.open(this.href, '_blank', 'width=1400,height=900'); return false;">
          Open Full Report
       </a>
   </div>

   <div class="benchmark-stats">
       <div class="stat-card">
           <div class="stat-number">30</div>
           <div class="stat-label">Total Tests</div>
       </div>
       <div class="stat-card">
           <div class="stat-number">4</div>
           <div class="stat-label">Categories</div>
       </div>
       <div class="stat-card">
           <div class="stat-number">3</div>
           <div class="stat-label">Vector Stores</div>
       </div>
       <div class="stat-card">
           <div class="stat-number">10K</div>
           <div class="stat-label">Max Vectors</div>
       </div>
   </div>

   <div class="info-note">
       <p style="margin: 0;">
           <strong>Note:</strong> The report opens in a new window. 
           If needed, navigate to <code>docs/_build/html/_static/benchmark_report.html</code> and open it directly.
       </p>
   </div>

Test Categories
---------------

.. raw:: html

   <div class="simple-grid">
       <div class="simple-card">
           <h4>📥 Insert Operations (9 tests)</h4>
           <ul>
               <li>Single vector insertion</li>
               <li>Small batch (100 vectors)</li>
               <li>Large batch (1K vectors)</li>
           </ul>
       </div>
       <div class="simple-card">
           <h4>🔍 Search Performance (9 tests)</h4>
           <ul>
               <li>Top-k similarity search</li>
               <li>High-volume queries</li>
               <li>Different dataset sizes</li>
           </ul>
       </div>
       <div class="simple-card">
           <h4>⚡ Batch Processing (6 tests)</h4>
           <ul>
               <li>Combined insert cycles</li>
               <li>Large-scale deletions</li>
               <li>Memory efficiency</li>
           </ul>
       </div>
       <div class="simple-card">
           <h4>📈 Scalability (6 tests)</h4>
           <ul>
               <li>10K+ vector operations</li>
               <li>Concurrent processing</li>
               <li>High-volume scenarios</li>
           </ul>
       </div>
   </div>

Vector Stores Tested
--------------------

.. raw:: html

   <div class="simple-grid">
       <div class="simple-card">
           <h4>🚀 Milvus</h4>
           <ul>
               <li>INT64 ID system</li>
               <li>Expression-based deletion</li>
               <li>Best for: Read-heavy workloads</li>
           </ul>
       </div>
       <div class="simple-card">
           <h4>⚡ Qdrant</h4>
           <ul>
               <li>UUID-based identifiers</li>
               <li>Batch processing support</li>
               <li>Best for: Single inserts (~1ms)</li>
           </ul>
       </div>
       <div class="simple-card">
           <h4>🎨 ChromaDB</h4>
           <ul>
               <li>In-memory operations</li>
               <li>Flexible metadata</li>
               <li>Best for: Development/testing</li>
           </ul>
       </div>
   </div>

How to Generate Fresh Results
------------------------------

.. code-block:: bash

   # Run complete benchmark suite (30-40 minutes)
   make benchmark
   
   # Generate interactive HTML report
   make benchmark-report
   
   # Update documentation with latest results
   make benchmark-docs
   
   # Rebuild Sphinx documentation
   cd docs && make html

Performance Highlights
----------------------

**Qdrant** - Best for single inserts (~1ms), excellent batch performance

**Milvus** - Fastest search operations, ideal for read-heavy workloads

**ChromaDB** - Balanced performance, perfect for development/testing

.. seealso::

   :doc:`benchmarks`
      Complete benchmark documentation with architecture details and usage guide.
