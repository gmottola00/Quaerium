// Custom JavaScript for RAG Toolkit docs

// Initialize Chart.js for benchmark visualizations
document.addEventListener('DOMContentLoaded', function() {
  // Remove target="_blank" from internal links to prevent opening new tabs
  const links = document.querySelectorAll('a');
  
  links.forEach(link => {
    const href = link.getAttribute('href');
    
    // Check if it's an internal link (relative or same domain)
    if (href && !href.startsWith('http') && !href.startsWith('//')) {
      // Remove target attribute to open in same tab
      link.removeAttribute('target');
    }
    
    // Only external links should open in new tab
    if (href && href.startsWith('http') && 
        !href.includes('localhost') && 
        !href.includes(window.location.hostname)) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });

  // Add smooth scroll behavior
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add copy feedback for code blocks
  document.querySelectorAll('.md-clipboard').forEach(button => {
    button.addEventListener('click', function() {
      const icon = this.querySelector('svg');
      if (icon) {
        icon.style.color = '#38ef7d';
        setTimeout(() => {
          icon.style.color = '';
        }, 2000);
      }
    });
  });

  // Add reading time estimate
  const content = document.querySelector('.md-content__inner');
  if (content) {
    const text = content.textContent;
    const wordCount = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / 200); // 200 words per minute
    
    if (readingTime > 1) {
      const timeElem = document.createElement('div');
      timeElem.className = 'reading-time';
      timeElem.innerHTML = `<small>📖 ${readingTime} min read</small>`;
      timeElem.style.cssText = 'color: var(--md-default-fg-color--light); margin-bottom: 1rem;';
      
      const title = content.querySelector('h1');
      if (title) {
        title.parentNode.insertBefore(timeElem, title.nextSibling);
      }
    }
  }

  // Progressive image loading
  const images = document.querySelectorAll('img[data-src]');
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.removeAttribute('data-src');
          imageObserver.unobserve(img);
        }
      });
    });

    images.forEach(img => imageObserver.observe(img));
  }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Ctrl/Cmd + K for search
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
      searchInput.focus();
    }
  }
});

// Analytics events (if Google Analytics is configured)
if (typeof gtag !== 'undefined') {
  // Track code copy events
  document.querySelectorAll('.md-clipboard').forEach(button => {
    button.addEventListener('click', function() {
      gtag('event', 'copy_code', {
        'event_category': 'engagement',
        'event_label': 'code_snippet'
      });
    });
  });

  // Track external link clicks
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    link.addEventListener('click', function() {
      gtag('event', 'click', {
        'event_category': 'outbound',
        'event_label': this.href
      });
    });
  });
}

// Add table of contents scroll spy
window.addEventListener('scroll', function() {
  const tocLinks = document.querySelectorAll('.md-nav--secondary a');
  let current = '';
  
  document.querySelectorAll('h2[id], h3[id]').forEach(section => {
    const sectionTop = section.offsetTop;
    if (scrollY >= sectionTop - 100) {
      current = section.getAttribute('id');
    }
  });

  tocLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === '#' + current) {
      link.classList.add('active');
    }
  });
});
