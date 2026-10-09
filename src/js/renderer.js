export class Renderer {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.config = config;
    this.resize();
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = this.config.width * ratio;
    this.canvas.height = this.config.height * ratio;
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  draw(game) {
    const ctx = this.context;
    const { width, height, birdX, birdRadius, pipeWidth, pipeGap } = this.config;
    ctx.fillStyle = '#0c1d23';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = '#1c3639';
    ctx.lineWidth = 1;
    for (let x = 24; x < width; x += 32) {
      for (let y = 24; y < height; y += 32) {
        ctx.beginPath(); ctx.moveTo(x - 2, y); ctx.lineTo(x + 2, y);
        ctx.moveTo(x, y - 2); ctx.lineTo(x, y + 2); ctx.stroke();
      }
    }
    this.drawCloud(75, 115, 0.85);
    this.drawCloud(355, 185, 1.15);
    this.drawCloud(240, 62, 0.6);

    const offset = (game.elapsedSeconds * 22) % width;
    ctx.fillStyle = '#173b35';
    ctx.beginPath(); ctx.moveTo(0, height);
    for (let x = 0; x <= width; x += 8) {
      ctx.lineTo(x, height - 57 + Math.sin((x + offset) / 65) * 20);
    }
    ctx.lineTo(width, height); ctx.fill();

    for (const pipe of game.pipes) {
      const top = pipe.gapCenterY - pipeGap / 2;
      const bottom = pipe.gapCenterY + pipeGap / 2;
      this.drawPipe(pipe.x, 0, pipeWidth, top, true);
      this.drawPipe(pipe.x, bottom, pipeWidth, height - bottom, false);
    }

    ctx.save();
    ctx.translate(birdX, game.bird.y);
    ctx.rotate(Math.max(-0.38, Math.min(0.85, game.bird.velocityY / 850)));
    ctx.fillStyle = '#c8a137';
    ctx.beginPath(); ctx.ellipse(-2, 5, birdRadius + 1, birdRadius, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#f8cb50';
    ctx.beginPath(); ctx.ellipse(0, 0, birdRadius + 2, birdRadius, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#604c26'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#e9ab34';
    ctx.beginPath(); ctx.ellipse(-7, 4, 8, 5, -0.3, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fffdf1';
    ctx.beginPath(); ctx.arc(6, -5, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#253a34';
    ctx.beginPath(); ctx.arc(8, -5, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e77747';
    ctx.beginPath(); ctx.moveTo(12, 0); ctx.lineTo(23, 3); ctx.lineTo(12, 7); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  drawCloud(x, y, scale) {
    const ctx = this.context;
    ctx.save(); ctx.translate(x, y); ctx.scale(scale, scale);
    ctx.fillStyle = '#21433f';
    ctx.beginPath(); ctx.roundRect(-40, 0, 90, 22, 11); ctx.fill();
    ctx.beginPath(); ctx.arc(-12, 0, 19, 0, Math.PI * 2); ctx.arc(15, -5, 25, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  drawPipe(x, y, width, height, top) {
    const ctx = this.context;
    const gradient = ctx.createLinearGradient(x, 0, x + width, 0);
    gradient.addColorStop(0, '#23534a'); gradient.addColorStop(0.18, '#428d6d');
    gradient.addColorStop(0.8, '#2f775f'); gradient.addColorStop(1, '#1d493f');
    ctx.fillStyle = gradient; ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = '#77c79a'; ctx.lineWidth = 2; ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = '#77c79a'; ctx.fillRect(x + 7, y, 5, height);
    const capY = top ? y + height - 22 : y;
    ctx.fillStyle = '#3f8e6c'; ctx.fillRect(x, capY, width, 22);
    ctx.strokeRect(x, capY, width, 22);
    ctx.fillStyle = '#a5e2b8'; ctx.fillRect(x + 2, capY + 2, width - 4, 4);
  }
}
