/**
 * BiBo vs Qwen3MoE — Report Interactive Logic
 * Handles navigation, tab switching, and image swapping.
 */

document.addEventListener('DOMContentLoaded', () => {
  // === Section Navigation ===
  const navBtns = document.querySelectorAll('.nav-btn');
  const sections = document.querySelectorAll('.section');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.section;
      
      // Update nav
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update sections
      sections.forEach(s => s.classList.remove('active'));
      document.getElementById(target).classList.add('active');
      
      // Scroll to top of section
      window.scrollTo({ top: document.querySelector('.nav').offsetTop, behavior: 'smooth' });
    });
  });

  // === Tab Image Switching ===
  // Each .tabs container controls the nearest .tab-img sibling
  document.querySelectorAll('.tabs').forEach(tabGroup => {
    const tabs = tabGroup.querySelectorAll('.tab');
    // Find the associated image — next sibling img with class tab-img
    const container = tabGroup.parentElement;
    const img = container.querySelector('.tab-img');
    
    if (!img) return;

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const imgName = tab.dataset.img;
        if (!imgName) return;
        
        // Update active tab
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update image
        img.src = `./plots/${imgName}.png`;
        img.alt = imgName;
        
        // Fade effect
        img.style.opacity = '0.5';
        img.onload = () => { img.style.opacity = '1'; };
      });
    });
  });

  // === Keyboard navigation ===
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const activeBtn = document.querySelector('.nav-btn.active');
      const allBtns = [...navBtns];
      const idx = allBtns.indexOf(activeBtn);
      
      let nextIdx;
      if (e.key === 'ArrowRight') {
        nextIdx = (idx + 1) % allBtns.length;
      } else {
        nextIdx = (idx - 1 + allBtns.length) % allBtns.length;
      }
      allBtns[nextIdx].click();
    }
  });

  // === Image zoom on click ===
  document.querySelectorAll('.plot-full img, .side img, .seq-selector img, .tab-img').forEach(img => {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', () => {
      // Create overlay
      const overlay = document.createElement('div');
      overlay.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.85); z-index: 9999;
        display: flex; align-items: center; justify-content: center;
        cursor: zoom-out; padding: 2rem;
      `;
      
      const bigImg = document.createElement('img');
      bigImg.src = img.src;
      bigImg.style.cssText = 'max-width: 95%; max-height: 95%; border-radius: 8px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);';
      
      overlay.appendChild(bigImg);
      document.body.appendChild(overlay);
      
      overlay.addEventListener('click', () => overlay.remove());
      document.addEventListener('keydown', function handler(e) {
        if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', handler); }
      });
    });
  });
});
