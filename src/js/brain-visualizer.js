// Presentation only: geometry and light levels consume already computed CSS values.
const WORLD = { width: 440, height: 424 };
const COLORS = { input: [103, 220, 236], hidden: [145, 174, 255], output: [248, 202, 112] };
const limit = (value) => Math.max(0, Math.min(1, value));
const rgba = (color, alpha) => `rgba(${color.join(',')},${alpha})`;

export class BrainVisualizer {
  constructor(canvas, status, architecture = [6, 16, 1]) {
    this.architecture = architecture;
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.status = status;
    this.visible = false;
    this.enabled = false;
    this.phase = 0;
    this.lastPaint = null;
    this.lastTime = null;
    this.needsDraw = true;
    this.lastPrediction = undefined;
    this.nextDischargeAt = 0;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.inputs = Array.from({ length: architecture[0] }, (_, i) => ({ x: 62, y: 65 + i * 288 / (architecture[0] - 1),
      label: ['Y', 'VY', 'DX', 'GAP', 'SPD', 'GAP2'][i], kind: 'input', level: 0, value: null, id: i }));
    this.hidden = Array.from({ length: architecture[1] }, (_, i) => ({ x: architecture[1] === 16 ? 197 + Math.floor(i / 8) * 56 : 225,
      y: 70 + (i % 8) * 41, label: `H${i + 1}`, kind: 'hidden', level: 0, value: null, id: i + architecture[0] }));
    this.output = { x: 380, y: 216, label: 'JUMP', kind: 'output', level: 0, value: null, id: architecture[0] + architecture[1] };
    this.edges = [
      ...this.inputs.flatMap((source) => this.hidden.map((target) => ({ source, target, color: COLORS.input }))),
      ...this.hidden.map((source) => ({ source, target: this.output, color: COLORS.hidden })),
    ];
    this.edges.forEach((edge) => {
      const { source, target } = edge;
      const bend = (target.x - source.x) * 0.48;
      edge.path = new Path2D();
      edge.path.moveTo(source.x, source.y);
      edge.path.bezierCurveTo(source.x + bend, source.y, target.x - bend, target.y, target.x, target.y);
      edge.points = Array.from({ length: 41 }, (_, i) => {
        const t = i / 40, u = 1 - t;
        return { x: u ** 3 * source.x + 3 * u * u * t * (source.x + bend) + 3 * u * t * t * (target.x - bend) + t ** 3 * target.x,
          y: u ** 3 * source.y + 3 * u * u * t * source.y + 3 * u * t * t * target.y + t ** 3 * target.y };
      });
      edge.discharge = null;
    });
    this.nodes = [...this.inputs, ...this.hidden, this.output];
    this.hidden.forEach((node) => {
      // Irregular cell bodies and branched dendrites, without inventing graph nodes.
      const outline = Array.from({ length: 14 }, (_, i) => {
        const angle = i / 14 * Math.PI * 2;
        const radius = 12 * (1 + 0.18 * Math.sin(i * 2.7 + node.id));
        return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
      });
      node.body = new Path2D();
      node.body.moveTo((outline[13].x + outline[0].x) / 2, (outline[13].y + outline[0].y) / 2);
      outline.forEach((point, i) => {
        const next = outline[(i + 1) % outline.length];
        node.body.quadraticCurveTo(point.x, point.y, (point.x + next.x) / 2, (point.y + next.y) / 2);
      });
      node.body.closePath();
      node.dendrites = new Path2D();
      for (let arm = 0; arm < 8; arm++) {
        const angle = arm * Math.PI / 4 + node.id * 0.41;
        const length = 19 + 5 * Math.sin(arm * 1.9 + node.id);
        const tip = { x: Math.cos(angle) * length, y: Math.sin(angle) * length };
        node.dendrites.moveTo(Math.cos(angle) * 9, Math.sin(angle) * 9);
        node.dendrites.lineTo(Math.cos(angle + 0.15) * 15, Math.sin(angle + 0.15) * 15);
        node.dendrites.lineTo(tip.x, tip.y);
        for (const turn of [-0.6, 0.65]) {
          node.dendrites.moveTo(tip.x, tip.y);
          node.dendrites.lineTo(tip.x + Math.cos(angle + turn) * 6, tip.y + Math.sin(angle + turn) * 6);
        }
      }
    });
    this.observer = new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.lastTime = null;
      this.lastPaint = null;
      this.needsDraw = true;
    });
    this.observer.observe(canvas);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.reducedMotion.addEventListener('change', () => {
      this.needsDraw = true;
    });
    this.resize();
  }

  resize() {
    const width = this.canvas.getBoundingClientRect().width;
    if (width === 0) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const height = width * WORLD.height / WORLD.width;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.context.setTransform(this.canvas.width / WORLD.width, 0, 0, this.canvas.height / WORLD.height, 0, 0);
    this.edges.forEach((edge) => {
      edge.gradient = this.context.createLinearGradient(edge.source.x, edge.source.y, edge.target.x, edge.target.y);
      edge.gradient.addColorStop(0, rgba(COLORS[edge.source.kind], 1));
      edge.gradient.addColorStop(0.55, rgba(COLORS.hidden, 1));
      edge.gradient.addColorStop(1, rgba(COLORS[edge.target.kind], 1));
    });
    this.background = null;
    this.needsDraw = true;
  }

  update(prediction, enabled, gameStatus) {
    if (this.enabled !== enabled || this.gameStatus !== gameStatus) this.needsDraw = true;
    this.enabled = enabled;
    this.gameStatus = gameStatus;
    if (prediction !== this.lastPrediction) {
      this.lastPrediction = prediction;
      this.inputs.forEach((node, index) => {
        node.value = prediction?.inputs[index] ?? null;
        node.level = node.value === null ? 0 : limit(Math.abs(node.value));
      });
      this.hidden.forEach((node, index) => {
        node.value = prediction?.hidden[index] ?? null;
        node.level = node.value === null ? 0 : limit(node.value / 4);
      });
      this.output.value = prediction?.probability ?? null;
      this.output.level = this.output.value ?? 0;
      if (!prediction) {
        this.edges.forEach(edge => edge.discharge = null);
        this.nodes.forEach(node => node.sparkAt = undefined);
        this.nextDischargeAt = this.phase;
      } else if (enabled && this.visible && !document.hidden && gameStatus === 'running' && !this.reducedMotion.matches) {
        this.discharge();
      }
      this.needsDraw = true;
    }
    const text = !prediction ? 'WAITING FOR AI FLIGHT' : gameStatus === 'running' ? 'LIVE · 20 DECISIONS/S' : gameStatus === 'paused' ? 'PAUSED' : 'FLIGHT COMPLETED';
    if (this.status.textContent !== text) this.status.textContent = text;
  }

  draw(timestamp) {
    if (!this.enabled || !this.visible || document.hidden) {
      this.lastTime = null;
      return;
    }
    const moving = !this.reducedMotion.matches && this.gameStatus === 'running';
    // The network is decorative; cap its paint rate without changing game or AI timing.
    if (moving && this.lastPaint !== null && timestamp - this.lastPaint < 1000 / 30) return;
    if (moving && this.lastTime !== null) this.phase += Math.min((timestamp - this.lastTime) / 1000, 0.1);
    this.lastTime = timestamp;
    if (!moving && !this.needsDraw) return;
    this.needsDraw = false;
    this.lastPaint = timestamp;
    const ctx = this.context;
    ctx.clearRect(0, 0, WORLD.width, WORLD.height);
    this.drawBackground();
    this.edges.forEach((edge) => this.drawEdge(edge, moving));
    this.nodes.forEach((node) => this.drawNode(node, this.phase));
    ctx.fillStyle = '#7faba3';
    ctx.font = '11px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`INPUTS / ${String(this.architecture[0]).padStart(2, '0')}`, 62, 27);
    ctx.fillText(`NEURONS / ${String(this.architecture[1]).padStart(2, '0')}`, 225, 27);
    ctx.fillText('OUTPUT / 01', 380, 27);
    ctx.fillStyle = '#7a9a87';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('STATE', 62, 414);
    ctx.fillText('ACTIVITY', 225, 414);
    ctx.fillText('DECISION', 380, 414);
  }

  drawBackground() {
    const ctx = this.context;
    if (!this.background) {
      this.background = document.createElement('canvas');
      this.background.width = this.canvas.width;
      this.background.height = this.canvas.height;
      const bg = this.background.getContext('2d');
      bg.setTransform(this.canvas.width / WORLD.width, 0, 0, this.canvas.height / WORLD.height, 0, 0);
      bg.fillStyle = '#080e1a';
      bg.fillRect(0, 0, WORLD.width, WORLD.height);
      const halo = bg.createRadialGradient(220, 210, 8, 220, 210, 250);
      halo.addColorStop(0, 'rgba(66, 63, 135, 0.14)');
      halo.addColorStop(0.55, 'rgba(40, 91, 112, 0.08)');
      halo.addColorStop(1, 'rgba(8, 14, 26, 0)');
      bg.fillStyle = halo;
      bg.fillRect(0, 0, WORLD.width, WORLD.height);
      bg.strokeStyle = 'rgba(125, 151, 193, 0.10)';
      bg.lineWidth = 0.6;
      this.edges.forEach(edge => bg.stroke(edge.path));
    }
    ctx.drawImage(this.background, 0, 0, WORLD.width, WORLD.height);
  }

  discharge() {
    if (this.phase < this.nextDischargeAt) return;
    this.nextDischargeAt = this.phase + 0.20 + Math.random() * 0.16;
    const active = this.hidden.filter(node => node.level > 0.005);
    if (!active.length) return;
    // Select a small cluster by real activity, not every connection on every decision.
    let choice = Math.random() * active.reduce((sum, node) => sum + Math.sqrt(node.level), 0);
    const neuron = active.find(node => (choice -= Math.sqrt(node.level)) <= 0) || active.at(-1);
    const incoming = this.edges.filter(edge => edge.target === neuron && edge.source.level > 0.01)
      .map(edge => ({ edge, priority: Math.random() * edge.source.level }))
      .sort((a, b) => b.priority - a.priority).slice(0, 2);
    incoming.forEach(({ edge }) => this.ignite(edge, 0));
    const outgoing = this.edges.find(edge => edge.source === neuron);
    if (outgoing) this.ignite(outgoing, 0.045);
    neuron.sparkAt = this.phase + 0.045;
  }

  ignite(edge, delay) {
    const path = new Path2D();
    const forks = new Path2D();
    const points = [];
    // Jagged, branching discharges stay close to the actual synaptic connection.
    for (let i = 0; i <= 12; i++) {
      const point = edge.points[Math.round(i / 12 * 40)];
      const jitter = Math.sin(i / 12 * Math.PI) * 9;
      points.push({ x: point.x + (Math.random() - 0.5) * jitter,
        y: point.y + (Math.random() - 0.5) * jitter * 1.7 });
    }
    points.forEach((point, i) => i ? path.lineTo(point.x, point.y) : path.moveTo(point.x, point.y));
    for (const index of [3, 7, 9]) {
      const root = points[index];
      const side = Math.random() < 0.5 ? -1 : 1;
      const x = root.x + 5 + Math.random() * 7, y = root.y + side * (5 + Math.random() * 7);
      forks.moveTo(root.x, root.y); forks.lineTo(x, y);
      forks.lineTo(x + 6, y + side * 4);
      forks.moveTo(x, y); forks.lineTo(x + 3, y - side * 4);
    }
    edge.discharge = { path, forks, started: this.phase + delay };
    edge.source.sparkAt = this.phase + delay;
    edge.target.sparkAt = this.phase + delay;
  }

  drawEdge(edge, moving) {
    const ctx = this.context;
    const { source, target, gradient, discharge } = edge;
    // Activation intensity is a visual cue, not a connection-weight estimate.
    const strength = Math.sqrt(target.kind === 'hidden' ? target.level * source.level : source.level);
    if (strength < 0.005) return;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = gradient;
    ctx.globalAlpha = strength * 0.14;
    ctx.lineWidth = 0.8;
    ctx.stroke(edge.path);
    const age = discharge ? this.phase - discharge.started : Infinity;
    if (moving && age >= 0 && age < 0.30) {
      const energy = Math.min(1, Math.sqrt(strength) * 1.9) * Math.min(1, age / 0.025) * Math.exp(-age / 0.12);
      // Local thin flashes; no full-panel strobe or moving particles.
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.globalAlpha = energy * 0.17; ctx.lineWidth = 8; ctx.stroke(discharge.path);
      ctx.globalAlpha = energy * 0.85; ctx.lineWidth = 2.4; ctx.stroke(discharge.path);
      ctx.globalAlpha = energy * 0.65; ctx.lineWidth = 0.7; ctx.stroke(discharge.forks);
      ctx.strokeStyle = '#e9faff';
      ctx.globalAlpha = energy; ctx.lineWidth = 0.9; ctx.stroke(discharge.path);
    } else if (!moving) {
      ctx.globalAlpha = strength * 0.26; ctx.lineWidth = 1.2; ctx.stroke(edge.path);
    }
    ctx.restore();
  }

  drawNode(node, time) {
    const ctx = this.context;
    const isOutput = node.kind === 'output';
    const color = COLORS[node.kind];
    const radius = node.kind === 'output' ? 24 : node.kind === 'hidden' ? 12 : 15;
    // The output is an instant readout, never a rotating or filling indicator.
    const breath = isOutput ? 1 : 1 + Math.sin(time * 1.3 + node.id * 0.7) * 0.025 * node.level;
    const level = isOutput ? 0.65 : Math.sqrt(node.level);
    const age = node.sparkAt === undefined ? Infinity : time - node.sparkAt;
    const charge = !isOutput && node.level > 0 && !this.reducedMotion.matches && this.gameStatus === 'running' && age >= 0 ? Math.exp(-age / 0.12) : 0;
    ctx.save(); ctx.translate(node.x, node.y);
    const bloom = ctx.createRadialGradient(0, 0, radius * 0.5, 0, 0, radius * 2.15);
    bloom.addColorStop(0, rgba(color, 0.025 + level * 0.18 + charge * 0.22));
    bloom.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = bloom; ctx.fillRect(-radius * 2.15, -radius * 2.15, radius * 4.3, radius * 4.3);

    // Fine dendrite branches decorate the actual graph nodes, not additional neurons.
    ctx.strokeStyle = rgba(color, 0.10 + level * 0.35 + charge * 0.5);
    ctx.lineWidth = 0.7;
    if (node.dendrites) ctx.stroke(node.dendrites);
    else for (let arm = 0; arm < 6; arm++) {
      const angle = arm * Math.PI / 3 + node.id * 0.47;
      const inner = radius * 0.88;
      const outer = radius * (1.38 + (arm % 2) * 0.17);
      const x = Math.cos(angle) * outer, y = Math.sin(angle) * outer;
      ctx.beginPath(); ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
      ctx.lineTo(x, y); ctx.lineTo(x + Math.cos(angle + 0.7) * 5, y + Math.sin(angle + 0.7) * 5); ctx.stroke();
    }
    if (isOutput) {
      ctx.strokeStyle = rgba(color, 0.35);
      ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.arc(0, 0, radius * 1.24, 0, Math.PI * 2); ctx.stroke();
    }

    const orb = ctx.createRadialGradient(-radius * 0.32, -radius * 0.38, 1, 0, 0, radius);
    orb.addColorStop(0, rgba(color, Math.min(1, 0.2 + level * 0.6 + charge * 0.35)));
    orb.addColorStop(0.42, rgba(color, 0.17 + level * 0.45));
    orb.addColorStop(1, '#10172b');
    ctx.fillStyle = orb; ctx.strokeStyle = rgba(color, Math.min(1, 0.25 + level * 0.55 + charge * 0.4));
    ctx.lineWidth = 1;
    if (node.body) { ctx.fill(node.body); ctx.stroke(node.body); }
    else { ctx.beginPath(); ctx.arc(0, 0, radius * breath, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = '#e1f3e8';
    ctx.font = '11px Consolas, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(node.label, 0, 0);
    if (node.kind !== 'hidden') {
      if (isOutput) {
        ctx.fillStyle = '#15241b';
        ctx.fillRect(-43, radius + 9, 86, 36);
        ctx.strokeStyle = rgba(color, 0.35);
        ctx.strokeRect(-43, radius + 9, 86, 36);
      }
      ctx.fillStyle = rgba(color, 0.8);
      ctx.font = node.kind === 'output' ? 'bold 18px Consolas, monospace' : '12px Consolas, monospace';
      const value = node.value === null ? '—' : node.kind === 'output' ? `${(node.value * 100).toFixed(1)}%` : node.value.toFixed(2);
      // Keep six-input values in the gap below their own node, above the next circle.
      ctx.fillText(value, 0, radius + (isOutput ? 27 : 14));
    }
    ctx.restore();
  }
}
