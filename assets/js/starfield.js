/*
  Home-page night sky (WebGL1, no dependencies).

  Two point clouds drawn with additive blending onto a transparent canvas that
  sits behind the profile:
    - sky:    stars on a unit sphere around the camera, slowly rotating about a
              tilted axis (a planetarium drift);
    - galaxy: a tilted two-armed spiral centred behind the avatar, with
              differential rotation (inner orbits faster) and perspective, so
              near particles are larger and move faster.
  Every star twinkles on its own phase; the pointer adds a small parallax tilt.

  Only runs in dark mode. Light mode, missing WebGL, or a lost context fall back
  to the CSS stars in home.css. prefers-reduced-motion draws one still frame.
*/
(() => {
  const profile = document.querySelector('.profile');
  if (!profile) return;

  const root = document.documentElement;
  const avatar = profile.querySelector('img');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const canvas = document.createElement('canvas');
  canvas.className = 'starfield';
  canvas.setAttribute('aria-hidden', 'true');
  const gl = canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    depth: false,
    powerPreference: 'low-power',
  });
  if (!gl) return;
  document.body.prepend(canvas);

  // ── Shaders ────────────────────────────────────────────────────────────
  const VERT = `
    precision highp float;
    attribute vec4 a_p;     // sky: dir.xyz, 0      galaxy: r, theta0, y, 1
    attribute vec4 a_c;     // rgb (pre-scaled by brightness), size in CSS px
    attribute vec2 a_t;     // twinkle speed, phase
    uniform float u_time;
    uniform float u_aspect;
    uniform float u_dpr;
    uniform vec2  u_center; // avatar centre, clip space
    uniform vec2  u_tilt;   // smoothed pointer offset, [-1, 1]
    varying vec3  v_col;
    varying float v_alpha;
    varying float v_soft;   // 1 for large nebula particles: no hard core
    varying float v_spike;  // 1 for the brightest sky stars: diffraction cross

    mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1., 0., 0.,  0., c, s,  0., -s, c); }
    mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c, 0., -s,  0., 1., 0.,  s, 0., c); }

    const float SKY_TILT = 0.42;
    const float GAL_TILT = 0.5;   // disk opens to an ellipse with ratio sin(0.5) ~ 0.48
    const float CAM_DIST = 2.4;
    const float FOCAL    = 2.5;

    void main() {
      float tw = 0.6 + 0.4 * sin(u_time * a_t.x + a_t.y);
      v_col = a_c.rgb;
      v_alpha = tw;
      v_soft = step(8., a_c.w);
      v_spike = (1. - a_p.w) * step(6., a_c.w);

      if (a_p.w < 0.5) {
        // Sky: rotate about an axis tilted by SKY_TILT, then a little parallax.
        vec3 d = rotX(SKY_TILT) * rotY(u_time * 0.01) * rotX(-SKY_TILT) * a_p.xyz;
        d = rotY(u_tilt.x * 0.04) * rotX(-u_tilt.y * 0.04) * d;
        if (d.z > -0.08) { gl_Position = vec4(2., 2., 2., 1.); gl_PointSize = 0.; return; }
        vec2 s = d.xy / -d.z * 0.9;
        gl_Position = vec4(s.x / u_aspect, s.y, 0., 1.);
        gl_PointSize = max(a_c.w * u_dpr, 1.5);
      } else {
        // Galaxy: Keplerian-ish differential rotation, then tilt and perspective.
        float r = a_p.x;
        float a = a_p.y + u_time * 0.05 / (0.22 + r);
        vec3 g = vec3(r * cos(a), a_p.z, r * sin(a));
        g = rotY(u_tilt.x * 0.18) * rotX(GAL_TILT + u_tilt.y * 0.12) * g;
        vec3 v = g - vec3(0., 0., CAM_DIST);
        float k = FOCAL / -v.z;
        gl_Position = vec4(v.x * k / u_aspect + u_center.x, v.y * k + u_center.y, 0., 1.);
        gl_PointSize = max(a_c.w * u_dpr * (CAM_DIST / -v.z), 1.5);
      }
    }`;

  const FRAG = `
    precision mediump float;
    varying vec3  v_col;
    varying float v_alpha;
    varying float v_soft;
    varying float v_spike;
    void main() {
      vec2 q = gl_PointCoord - 0.5;
      float d = length(q) * 2.;
      float core = 1. - smoothstep(0., 0.35, d);
      float halo = pow(max(1. - d, 0.), 3.) * 0.8;
      float soft = pow(max(1. - d, 0.), 2.);
      float spike = (exp(-abs(q.x) * 60.) + exp(-abs(q.y) * 60.)) * pow(max(1. - d, 0.), 1.5) * 0.7;
      float a = (mix(core + halo, soft, v_soft) + spike * v_spike) * v_alpha;
      // Premultiplied output; alpha follows brightness so faint particles only
      // add light instead of darkening the page behind them.
      float lum = max(max(v_col.r, v_col.g), v_col.b);
      gl_FragColor = vec4(v_col * a, a * lum);
    }`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  } catch (e) {
    canvas.remove();
    return;
  }
  gl.useProgram(prog);

  // ── Stars (seeded, so the sky looks the same on every visit) ──────────
  let seed = 0x9e3779b9;
  const rand = () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

  // Catppuccin Mocha
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const C = {
    white: hex('#f5f7ff'), rosewater: hex('#f5e0dc'), lavender: hex('#b4befe'),
    blue: hex('#89b4fa'), mauve: hex('#cba6f7'), pink: hex('#f5c2e7'), yellow: hex('#f9e2af'),
  };
  const pick = (table) => {
    let x = rand();
    for (const [c, w] of table) { if ((x -= w) <= 0) return c; }
    return table[0][0];
  };

  const data = [];
  const push = (p, rgb, brightness, size, speed, phase) =>
    data.push(...p, rgb[0] * brightness, rgb[1] * brightness, rgb[2] * brightness, size, speed, phase);

  // Star sizes follow a power law, like stellar magnitudes: most stars are faint
  // pin-pricks, a few are bright. m in [0, 1], 1 = brightest.
  const magnitude = (k) => Math.pow(rand(), k);

  const SKY = 2400;
  const skyColors = [[C.white, 0.62], [C.lavender, 0.2], [C.yellow, 0.1], [C.pink, 0.08]];
  for (let i = 0; i < SKY; i++) {
    const z = 2 * rand() - 1, phi = 2 * Math.PI * rand(), s = Math.sqrt(1 - z * z);
    const m = magnitude(7);
    push([s * Math.cos(phi), s * Math.sin(phi), z, 0], pick(skyColors),
      0.55 + 1.7 * m,              // brightness
      0.9 + 6.5 * m,               // size (CSS px); m > ~0.78 (about 3%) gets diffraction spikes
      0.6 + 2.4 * rand(), 2 * Math.PI * rand());
  }

  const GALAXY = 7000;
  for (let i = 0; i < GALAXY; i++) {
    const halo = rand() < 0.18;
    const r = Math.pow(rand(), 1.35);
    const arm = (i & 1) * Math.PI;
    const theta = halo ? 2 * Math.PI * rand() : arm + r * 4.4 + gauss() * (0.38 - 0.2 * r);
    const y = gauss() * 0.045 * (1 - 0.6 * r);
    const rr = r + gauss() * 0.015;
    const color = r < 0.12 ? pick([[C.rosewater, 0.6], [C.yellow, 0.4]])
      : r < 0.55 ? pick([[C.mauve, 0.45], [C.lavender, 0.35], [C.white, 0.1], [C.pink, 0.1]])
      : pick([[C.blue, 0.5], [C.lavender, 0.3], [C.white, 0.1], [C.pink, 0.1]]);
    const dust = rand() < 0.06;   // a few large, very faint particles: a smooth glow under the grain
    const m = magnitude(5);
    push([Math.max(rr, 0.005), theta, y, 1], color,
      dust ? 0.05 + 0.04 * rand() : (r < 0.12 ? 1.3 : 0.7) + 1.1 * m,
      dust ? 22 + 22 * rand() : 0.9 + 2.4 * m,
      0.4 + 1.8 * rand(), 2 * Math.PI * rand());
  }

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
  const STRIDE = 10 * 4;
  [['a_p', 4, 0], ['a_c', 4, 16], ['a_t', 2, 32]].forEach(([name, n, off]) => {
    const loc = gl.getAttribLocation(prog, name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, n, gl.FLOAT, false, STRIDE, off);
  });
  const COUNT = SKY + GALAXY;

  const U = {};
  ['u_time', 'u_aspect', 'u_dpr', 'u_center', 'u_tilt'].forEach((n) => (U[n] = gl.getUniformLocation(prog, n)));

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);   // additive: overlapping light adds up
  gl.clearColor(0, 0, 0, 0);

  // ── Layout, pointer, theme ─────────────────────────────────────────────
  let dpr = 1, center = [0, 0];
  function locate() {
    if (!avatar) return;
    const b = avatar.getBoundingClientRect();
    center = [((b.left + b.width / 2) / window.innerWidth) * 2 - 1,
              1 - ((b.top + b.height / 2) / window.innerHeight) * 2];
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    locate();
  }

  const tilt = [0, 0], target = [0, 0];
  window.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    target[0] = (e.clientX / window.innerWidth) * 2 - 1;
    target[1] = (e.clientY / window.innerHeight) * 2 - 1;
  }, { passive: true });

  const start = performance.now() - 40000;   // start mid-rotation, not at t = 0
  function draw(now) {
    tilt[0] += (target[0] - tilt[0]) * 0.04;
    tilt[1] += (target[1] - tilt[1]) * 0.04;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(U.u_time, (now - start) / 1000);
    gl.uniform1f(U.u_aspect, canvas.width / canvas.height);
    gl.uniform1f(U.u_dpr, dpr);
    gl.uniform2f(U.u_center, center[0], center[1]);
    gl.uniform2f(U.u_tilt, tilt[0], tilt[1]);
    gl.drawArrays(gl.POINTS, 0, COUNT);
  }

  let raf = 0, running = false;
  const loop = (now) => { draw(now); raf = requestAnimationFrame(loop); };

  function update() {
    const on = root.dataset.theme === 'dark' && !gl.isContextLost();
    root.classList.toggle('has-starfield', on);
    if (!on) { cancelAnimationFrame(raf); running = false; return; }
    resize();
    if (reduceMotion.matches) {
      cancelAnimationFrame(raf); running = false;
      draw(start + 40000);
    } else if (!running) {
      running = true;
      raf = requestAnimationFrame(loop);
    }
  }

  window.addEventListener('resize', update, { passive: true });
  window.addEventListener('scroll', locate, { passive: true });
  reduceMotion.addEventListener('change', update);
  new MutationObserver(update).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); update(); });
  
  update();
})();
