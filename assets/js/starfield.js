/*
  Site background (WebGL1, no dependencies), loaded on every page from
  layouts/partials/extend_footer.html.

  Dark — night sky (additive light):
    - sky:    stars on a unit sphere, slowly rotating about a tilted axis;
    - galaxy: a tilted two-armed spiral with differential rotation and
              perspective. On the home page it sits behind the avatar at full
              strength; elsewhere it moves to the upper right and dims so it
              never competes with the text column.
  Light — daylight aurora (normal blending):
    - a full-screen domain-warped noise field in Catppuccin Latte pastels,
      flowing very slowly, like watercolour;
    - pastel motes drifting upwards with a gentle sway; the largest glint.

  Star sizes follow a power law (many faint, few bright). Every point twinkles
  on its own phase; the pointer adds a small parallax tilt.
  Fallbacks: no WebGL or a lost context -> the CSS backgrounds in home.css;
  prefers-reduced-motion -> one still frame.
*/
(() => {
  const root = document.documentElement;
  const isHome = !!document.querySelector('.profile');
  const avatar = isHome ? document.querySelector('.profile img') : null;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const small = Math.min(window.innerWidth, window.innerHeight) < 700;

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
  const POINT_VERT = `
    precision highp float;
    attribute vec4 a_p;     // sky: dir.xyz, 0 | galaxy: r, theta0, y, 1 | mote: x, y, rise, 2
    attribute vec4 a_c;     // rgb (pre-scaled by brightness), size in CSS px
    attribute vec2 a_t;     // twinkle speed, phase
    uniform float u_time;
    uniform float u_aspect;
    uniform float u_dpr;
    uniform vec2  u_center;    // galaxy centre, clip space
    uniform float u_galScale;  // galaxy size on screen
    uniform vec3  u_gain;      // brightness: sky, galaxy, motes
    uniform vec2  u_tilt;      // smoothed pointer offset, [-1, 1]
    varying vec3  v_col;
    varying float v_alpha;
    varying float v_soft;      // large nebula particles: no hard core
    varying float v_spike;     // brightest stars / motes: a diffraction cross

    mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1., 0., 0.,  0., c, s,  0., -s, c); }
    mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c, 0., -s,  0., 1., 0.,  s, 0., c); }

    const float SKY_TILT = 0.42;
    const float GAL_TILT = 0.5;   // disk opens to an ellipse with ratio sin(0.5) ~ 0.48
    const float CAM_DIST = 2.4;
    const float FOCAL    = 2.5;

    void main() {
      float tw = 0.6 + 0.4 * sin(u_time * a_t.x + a_t.y);
      v_col = a_c.rgb;
      v_soft = step(8., a_c.w);
      v_spike = 0.;

      if (a_p.w < 0.5) {
        // Sky: rotate about an axis tilted by SKY_TILT, then a little parallax.
        vec3 d = rotX(SKY_TILT) * rotY(u_time * 0.01) * rotX(-SKY_TILT) * a_p.xyz;
        d = rotY(u_tilt.x * 0.04) * rotX(-u_tilt.y * 0.04) * d;
        if (d.z > -0.08) { gl_Position = vec4(2., 2., 2., 1.); gl_PointSize = 0.; return; }
        vec2 s = d.xy / -d.z * 0.9;
        gl_Position = vec4(s.x / u_aspect, s.y, 0., 1.);
        gl_PointSize = max(a_c.w * u_dpr, 1.5);
        v_alpha = tw * u_gain.x;
        v_spike = step(6., a_c.w);
      } else if (a_p.w < 1.5) {
        // Galaxy: Keplerian-ish differential rotation, then tilt and perspective.
        float r = a_p.x;
        float a = a_p.y + u_time * 0.05 / (0.22 + r);
        vec3 g = vec3(r * cos(a), a_p.z, r * sin(a));
        g = rotY(u_tilt.x * 0.18) * rotX(GAL_TILT + u_tilt.y * 0.12) * g;
        vec3 v = g - vec3(0., 0., CAM_DIST);
        float k = FOCAL / -v.z * u_galScale;
        gl_Position = vec4(v.x * k / u_aspect + u_center.x, v.y * k + u_center.y, 0., 1.);
        gl_PointSize = max(a_c.w * u_dpr * (CAM_DIST / -v.z) * mix(1., u_galScale, 0.5), 1.5);
        v_alpha = tw * u_gain.y;
      } else {
        // Motes: rise slowly with a sideways sway, wrapping top to bottom.
        float y = fract(a_p.y + u_time * a_p.z);
        float x = a_p.x + 0.012 * sin(u_time * 0.25 + a_t.y) + u_tilt.x * 0.01 * a_c.w / 6.;
        gl_Position = vec4(x * 2. - 1., y * 2.2 - 1.1, 0., 1.);
        gl_PointSize = max(a_c.w * u_dpr, 1.5);
        // Glints: mostly dim, with short bright flashes.
        float flash = 0.25 + 0.75 * pow(0.5 + 0.5 * sin(u_time * a_t.x + a_t.y), 4.);
        v_alpha = flash * u_gain.z * smoothstep(0., 0.12, y) * smoothstep(1., 0.85, y);
        v_spike = step(4.5, a_c.w);
      }
    }`;

  const POINT_FRAG = `
    precision mediump float;
    uniform float u_light;   // 0: additive night sky, 1: normal blending on a light page
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
      // Premultiplied output with alpha = coverage x brightness, so every
      // component stays <= alpha. At night (additive) faint particles only add
      // light; by day the pure hue is laid over the page at that opacity.
      float lum = max(max(v_col.r, v_col.g), v_col.b);
      float alpha = min(a * lum, 1.);
      vec3 rgb = mix(v_col * a, v_col / max(lum, 1e-3) * alpha, u_light);
      gl_FragColor = vec4(rgb, alpha);
    }`;

  const AURORA_VERT = `
    attribute vec2 a_xy;
    void main() { gl_Position = vec4(a_xy, 0., 1.); }`;

  const AURORA_FRAG = `
    precision mediump float;
    uniform vec2  u_res;
    uniform float u_time;
    uniform float u_gain;
    uniform vec2  u_tilt;

    float hash(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }
    float noise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      vec2 u = f * f * (3. - 2. * f);
      return mix(mix(hash(i), hash(i + vec2(1., 0.)), u.x),
                 mix(hash(i + vec2(0., 1.)), hash(i + vec2(1., 1.)), u.x), u.y);
    }
    float fbm(vec2 p) {
      float v = 0., a = 0.5;
      for (int i = 0; i < 4; i++) { v += a * noise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; }
      return v;
    }

    // Catppuccin Latte
    const vec3 MAUVE = vec3(0.533, 0.224, 0.937);
    const vec3 BLUE  = vec3(0.118, 0.400, 0.961);
    const vec3 PINK  = vec3(0.918, 0.463, 0.796);
    const vec3 TEAL  = vec3(0.090, 0.573, 0.600);
    const vec3 SAPPH = vec3(0.125, 0.624, 0.710);

    void main() {
      vec2 uv = gl_FragCoord.xy / u_res.y + u_tilt * 0.02;
      float t = u_time * 0.02;
      // Domain warping: noise displaced by noise gives slow, folding ribbons.
      vec2 q = vec2(fbm(uv * 1.2 + vec2(0., t)), fbm(uv * 1.2 + vec2(5.2, -t)));
      vec2 r = vec2(fbm(uv * 1.0 + 2.2 * q + vec2(1.7, 9.2) + t * 0.6),
                    fbm(uv * 1.0 + 2.2 * q + vec2(8.3, 2.8) - t * 0.4));
      float n = fbm(uv * 0.9 + 2.0 * r);

      vec3 col = mix(BLUE, MAUVE, smoothstep(0.15, 0.6, q.x));
      col = mix(col, PINK, smoothstep(0.4, 0.75, r.y) * 0.8);
      col = mix(col, TEAL, smoothstep(0.55, 0.9, r.x) * 0.45);
      col = mix(col, SAPPH, smoothstep(0.6, 0.95, n) * 0.3);

      // A light wash plus a few thin ribbons along iso-lines of the warped
      // field: crisp silk-like bands instead of a uniform fog.
      float fill = smoothstep(0.35, 0.8, n) * 0.35;
      float ribbons = exp(-pow((n - 0.52) / 0.012, 2.))
                    + 0.75 * exp(-pow((n - 0.62) / 0.01, 2.))
                    + 0.55 * exp(-pow((n - 0.43) / 0.011, 2.));
      float a = (fill + 0.7 * ribbons) * u_gain;
      gl_FragColor = vec4(col * a, a);
    }`;

  function program(vs, fs) {
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
  }

  let points, aurora;
  try {
    points = program(POINT_VERT, POINT_FRAG);
    aurora = program(AURORA_VERT, AURORA_FRAG);
  } catch (e) {
    canvas.remove();
    return;
  }

  // ── Geometry (seeded, so the sky looks the same on every visit) ────────
  let seed = 0x9e3779b9;
  const rand = () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const magnitude = (k) => Math.pow(rand(), k);   // power law: mostly faint, a few bright

  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const M = {   // Catppuccin Mocha
    white: hex('#f5f7ff'), rosewater: hex('#f5e0dc'), lavender: hex('#b4befe'),
    blue: hex('#89b4fa'), mauve: hex('#cba6f7'), pink: hex('#f5c2e7'), yellow: hex('#f9e2af'),
  };
  const L = {   // Catppuccin Latte
    mauve: hex('#8839ef'), blue: hex('#1e66f5'), lavender: hex('#7287fd'),
    pink: hex('#ea76cb'), sapphire: hex('#209fb5'), peach: hex('#fe640b'),
  };
  const pick = (table) => {
    let x = rand();
    for (const [c, w] of table) { if ((x -= w) <= 0) return c; }
    return table[0][0];
  };

  const data = [];
  const push = (p, rgb, brightness, size, speed, phase) =>
    data.push(...p, rgb[0] * brightness, rgb[1] * brightness, rgb[2] * brightness, size, speed, phase);

  const SKY = small ? 1400 : 2400;
  const skyColors = [[M.white, 0.62], [M.lavender, 0.2], [M.yellow, 0.1], [M.pink, 0.08]];
  for (let i = 0; i < SKY; i++) {
    const z = 2 * rand() - 1, phi = 2 * Math.PI * rand(), s = Math.sqrt(1 - z * z);
    const m = magnitude(7);
    push([s * Math.cos(phi), s * Math.sin(phi), z, 0], pick(skyColors),
      0.55 + 1.7 * m,     // brightness
      0.9 + 6.5 * m,      // size (CSS px); m > ~0.78 (about 3%) gets diffraction spikes
      0.6 + 2.4 * rand(), 2 * Math.PI * rand());
  }

  const GALAXY = small ? 3500 : 7000;
  for (let i = 0; i < GALAXY; i++) {
    const halo = rand() < 0.18;
    const r = Math.pow(rand(), 1.35);
    const arm = (i & 1) * Math.PI;
    const theta = halo ? 2 * Math.PI * rand() : arm + r * 4.4 + gauss() * (0.38 - 0.2 * r);
    const y = gauss() * 0.045 * (1 - 0.6 * r);
    const rr = r + gauss() * 0.015;
    const color = r < 0.12 ? pick([[M.rosewater, 0.6], [M.yellow, 0.4]])
      : r < 0.55 ? pick([[M.mauve, 0.45], [M.lavender, 0.35], [M.white, 0.1], [M.pink, 0.1]])
      : pick([[M.blue, 0.5], [M.lavender, 0.3], [M.white, 0.1], [M.pink, 0.1]]);
    const dust = rand() < 0.06;   // a few large, very faint particles: a smooth glow under the grain
    const m = magnitude(5);
    push([Math.max(rr, 0.005), theta, y, 1], color,
      dust ? 0.05 + 0.04 * rand() : (r < 0.12 ? 1.3 : 0.7) + 1.1 * m,
      dust ? 22 + 22 * rand() : 0.9 + 2.4 * m,
      0.4 + 1.8 * rand(), 2 * Math.PI * rand());
  }

  const MOTES = small ? 220 : 420;
  const moteColors = [[L.lavender, 0.3], [L.mauve, 0.25], [L.blue, 0.2], [L.pink, 0.15], [L.sapphire, 0.07], [L.peach, 0.03]];
  for (let i = 0; i < MOTES; i++) {
    const m = magnitude(4);
    push([rand(), rand(), 0.004 + 0.008 * rand(), 2], pick(moteColors),
      0.5 + 0.5 * m,      // opacity on the light page
      3 + 9 * m,          // size; the larger third (> 4.5) get a four-point glint
      0.5 + 1.5 * rand(), 2 * Math.PI * rand());
  }

  const pointBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const pointAttrs = [['a_p', 4, 0], ['a_c', 4, 16], ['a_t', 2, 32]]
    .map(([name, n, off]) => [gl.getAttribLocation(points.p, name), n, off]);
  const auroraAttr = gl.getAttribLocation(aurora.p, 'a_xy');

  function usePoints() {
    gl.useProgram(points.p);
    gl.disableVertexAttribArray(auroraAttr);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
    for (const [loc, n, off] of pointAttrs) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 40, off);
    }
  }
  function useAurora() {
    gl.useProgram(aurora.p);
    for (const [loc] of pointAttrs) gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(auroraAttr);
    gl.vertexAttribPointer(auroraAttr, 2, gl.FLOAT, false, 0, 0);
  }

  gl.enable(gl.BLEND);
  gl.clearColor(0, 0, 0, 0);

  // ── Per-page strength ──────────────────────────────────────────────────
  // Home is the showcase; on reading pages everything steps back.
  const look = isHome
    ? { sky: 1, galaxy: 1, galScale: 1, motes: 1, aurora: 0.3 }
    : { sky: 0.7, galaxy: 0.45, galScale: 0.7, motes: 0.65, aurora: 0.18 };

  // ── Layout, pointer, theme ─────────────────────────────────────────────
  let dpr = 1, center = [0.72, 0.5];   // off to the upper right unless there is an avatar
  function locate() {
    if (!avatar) return;
    const b = avatar.getBoundingClientRect();
    center = [((b.left + b.width / 2) / window.innerWidth) * 2 - 1,
              1 - ((b.top + b.height / 2) / window.innerHeight) * 2];
  }
  // The daylight aurora is slow: render it at up to 1.5x and 30 fps, which cuts
  // the per-pixel noise work by ~3.5x. The night sky needs crisp points: up to 2x.
  const isLight = () => root.dataset.theme !== 'dark';
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, isLight() ? 1.5 : 2);
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
    const light = isLight();
    const time = (now - start) / 1000;
    tilt[0] += (target[0] - tilt[0]) * 0.04;
    tilt[1] += (target[1] - tilt[1]) * 0.04;
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (light) {
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      useAurora();
      gl.uniform2f(aurora.u.u_res, canvas.width, canvas.height);
      gl.uniform1f(aurora.u.u_time, time);
      gl.uniform1f(aurora.u.u_gain, look.aurora);
      gl.uniform2f(aurora.u.u_tilt, tilt[0], tilt[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      gl.blendFunc(gl.ONE, gl.ONE);   // additive: overlapping light adds up
    }

    usePoints();
    gl.uniform1f(points.u.u_time, time);
    gl.uniform1f(points.u.u_aspect, canvas.width / canvas.height);
    gl.uniform1f(points.u.u_dpr, dpr);
    gl.uniform2f(points.u.u_center, center[0], center[1]);
    gl.uniform1f(points.u.u_galScale, look.galScale);
    gl.uniform3f(points.u.u_gain, look.sky, look.galaxy, look.motes);
    gl.uniform2f(points.u.u_tilt, tilt[0], tilt[1]);
    gl.uniform1f(points.u.u_light, light ? 1 : 0);
    if (light) gl.drawArrays(gl.POINTS, SKY + GALAXY, MOTES);
    else gl.drawArrays(gl.POINTS, 0, SKY + GALAXY);
  }

  let raf = 0, running = false, frame = 0;
  const loop = (now) => {
    if (!(isLight() && (frame++ & 1))) draw(now);
    raf = requestAnimationFrame(loop);
  };

  function update() {
    const on = !gl.isContextLost();
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
  // The theme toggle flips data-theme: resize for the new resolution (the loop
  // redraws next frame), or redraw the still frame under reduced motion.
  new MutationObserver(() => { if (running) resize(); else update(); })
    .observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); update(); });

  update();
})();
