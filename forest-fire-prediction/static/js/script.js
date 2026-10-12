document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));

// Animate the circular probability gauge
const gauge = document.querySelector('.gauge');
if (gauge) {
  const target = parseFloat(gauge.dataset.value), start = performance.now();
  const step = t => {
    const k = Math.min((t - start) / 800, 1);
    gauge.style.setProperty('--p', target * k);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
}

// Sample inputs only; the model produces the result
document.getElementById('fill-sample').addEventListener('click', () => {
  const s = { Temperature: 35, RH: 35, Ws: 15, Rain: 0, FFMC: 90, DMC: 40, DC: 150, ISI: 10, BUI: 45 };
  Object.entries(s).forEach(([k, v]) => document.getElementById(k).value = v);
});

document.getElementById('predict-form').addEventListener('submit', e => {
  if (!e.target.checkValidity()) { e.preventDefault(); e.target.reportValidity(); }
});
