const BASE = (window.APP_CONFIG && window.APP_CONFIG.BASE_URL) || '';

const byId = (id) => document.getElementById(id);

const uploadForm = byId('upload-form');
const predictForm = byId('predict-form');
const uploadStatus = byId('upload-status');
const resultBox = byId('result');

function setPredictEnabled(enabled) {
  if (enabled) {
    predictForm.classList.remove('disabled');
  } else {
    predictForm.classList.add('disabled');
  }
}

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  uploadStatus.textContent = '';
  uploadStatus.className = 'status';

  const modelFile = byId('modelFile').files[0];
  const metaFile = byId('metaFile').files[0];
  const classesFile = byId('classesFile').files[0];
  const scalerFile = byId('scalerFile').files[0];
  if (!modelFile || !metaFile || !classesFile || !scalerFile) {
    uploadStatus.textContent = 'Please select all four files.';
    uploadStatus.classList.add('error');
    return;
  }

  const formData = new FormData();
  formData.append('model', modelFile);
  formData.append('meta', metaFile);
  formData.append('classes', classesFile);
  formData.append('scaler', scalerFile);

  try {
    const res = await fetch(`${BASE}/upload`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    const flags = [];
    flags.push(data.has_scaler ? 'scaler: OK' : 'scaler: MISSING (using identity)');
    if (data.has_classes && data.classes) {
      flags.push(`classes: OK (${data.classes.join(', ')})`);
    } else {
      flags.push('classes: MISSING');
    }
    uploadStatus.textContent = `Model loaded. ${flags.join(' | ')}`;
    uploadStatus.classList.add('success');
    setPredictEnabled(true);
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.classList.add('error');
    setPredictEnabled(false);
  }
});

predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultBox.innerHTML = '';

  const payload = {
    age: parseFloat(byId('age').value),
    gender: byId('gender').value,
    bmi: parseFloat(byId('bmi').value),
    chol: parseFloat(byId('chol').value),
    tg: parseFloat(byId('tg').value),
    hdl: parseFloat(byId('hdl').value),
    ldl: parseFloat(byId('ldl').value),
    creatinine: parseFloat(byId('creatinine').value),
    bun: parseFloat(byId('bun').value),
  };

  try {
    const res = await fetch(`${BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Prediction failed');
    }
    const prob = data.probability !== undefined ? (data.probability * 100).toFixed(2) + '%' : '—';
    const dist = data.distribution ? `<div class="prob">Distribution: ${data.distribution.map(v => v.toFixed(4)).join(', ')}</div>` : '';
    
    // Show both probabilities for binary classification
    let probDetails = '';
    if (data.probabilities) {
      const probNo = (data.probabilities.no_diabetes * 100).toFixed(2);
      const probYes = (data.probabilities.diabetes * 100).toFixed(2);
      probDetails = `
        <div class="prob" style="margin-top: 8px;">
          <div>No Diabetes: ${probNo}%</div>
          <div>Diabetes: ${probYes}%</div>
        </div>
      `;
    }
    
    resultBox.innerHTML = `
      <div class="value">Prediction: ${data.prediction}</div>
      <div class="prob">Confidence: ${prob}</div>
      ${probDetails}
      ${dist}
    `;
  } catch (err) {
    resultBox.innerHTML = `<span style="color:#ef4444">${err.message}</span>`;
  }
});


