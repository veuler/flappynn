export function bindInput({ canvas, onJump, onPause }) {
  const keydown = (event) => {
    if (event.target.closest('input, select, textarea, button, a')) return;
    if (['Space', 'ArrowUp', 'KeyW'].includes(event.code)) {
      event.preventDefault();
      if (!event.repeat) onJump();
    } else if (event.code === 'KeyP' && !event.repeat) {
      event.preventDefault();
      onPause();
    }
  };
  document.addEventListener('keydown', keydown);
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    onJump();
  });
}
