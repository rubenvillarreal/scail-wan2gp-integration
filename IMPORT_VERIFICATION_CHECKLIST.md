# SCAIL Import Verification Checklist

## Quick Status Check

Run these commands to verify all critical dependencies:

```bash
# 1. Test core import chain
python -c "from wan2gp_integration.models.wan.any2video import WanAny2V; print('✅ Core imports OK')"

# 2. Verify regex package (critical for tokenizers)
python -c "import regex; print(f'✅ regex {regex.__version__} installed')"

# 3. Verify all critical packages
python << 'EOF'
import sys
critical_packages = [
    'torch', 'numpy', 'transformers', 'diffusers', 'accelerate',
    'mmgp', 'onnxruntime', 'cv2', 'PIL', 'einops', 'ftfy',
    'segment_anything', 'timm', 'insightface', 'taichi', 'smplfitter'
]

missing = []
for pkg in critical_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} MISSING')

if missing:
    print(f'\n⚠️  Missing packages: {", ".join(missing)}')
    sys.exit(1)
else:
    print('\n✅ All critical dependencies satisfied!')
EOF
```

## Expected Results

### ✅ All tests PASS:
- Core imports: OK
- regex: 2025.10.23 (or newer)
- All critical packages: ✅

### ❌ If any test FAILS:

#### Missing regex:
```bash
pip install regex>=2023.12.25
```

#### Missing other packages:
```bash
pip install -r requirements_wan2gp.txt
```

## Import Chain Reference

```
handler.py
  └─> wan2gp_integration/models/wan/any2video.py
      ├─> modules/model.py (WanModel)
      ├─> modules/t5.py (T5EncoderModel)
      │   └─> tokenizers.py (needs: regex, ftfy, transformers)
      ├─> modules/vae.py (WanVAE)
      ├─> shared/attention.py (optional: flash-attn, xformers, sageattention)
      └─> scail/model_scail.py
```

## Known Good Configurations

### Python Environment
- Python: 3.11+
- CUDA: 12.1+
- PyTorch: 2.3.1+

### Critical Dependencies (in requirements_wan2gp.txt)
- transformers==4.53.1
- diffusers==0.34.0
- mmgp==3.6.9
- onnxruntime-gpu==1.22
- All SCAIL pose dependencies

### Installed but not in requirements
- regex (2025.10.23) - Should be added for documentation

### Not needed for SCAIL
- matplotlib (only for non-SCAIL pose utilities)
- rembg (background removal, lazily loaded)

## Troubleshooting

### Issue: ModuleNotFoundError: regex
**Solution:** `pip install regex>=2023.12.25`

### Issue: ModuleNotFoundError: [package]
**Solution:** `pip install -r requirements_wan2gp.txt`

### Issue: Slow inference
**Optional optimization:** Install flash-attn or xformers for 10-20% speedup
```bash
pip install flash-attn>=2.0.0  # If GPU compatible
pip install xformers>=0.0.20   # Alternative
```

## Files Created

1. `IMPORT_AUDIT.txt` - Detailed technical analysis (all imports)
2. `IMPORT_AUDIT_SUMMARY.txt` - Executive summary (deployment readiness)
3. `IMPORT_VERIFICATION_CHECKLIST.md` - This file (quick reference)

## Last Updated
2025-12-22 - Full import chain audit completed
