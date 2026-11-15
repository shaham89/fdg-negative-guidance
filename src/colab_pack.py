"""Pack the code, the scripts and the small tables the code reads into freqcfg.zip, for a Google Colab runtime.
No samples and no checkpoints (scripts/setup.sh downloads those on the runtime).
  python src/colab_pack.py [out.zip]
Then drag the zip into Colab's Files panel and run scripts/colab.ipynb: the repository lands in /content/freqcfg."""
import os, sys, glob, zipfile

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(repo, *p)
dst = sys.argv[1] if len(sys.argv) > 1 else J('freqcfg.zip')
files = glob.glob(J('src', '*.py')) + glob.glob(J('scripts', '*.sh')) + [J('requirements.txt')] \
    + [J('results', 'two_dials', 'cifar_spectrum.npz'), J('results', 'two_dials', 'wiener_table.json')] + glob.glob(J('results', 'two_dials', 'time_from_*.json')) \
    + [J('results', 'neg', 'reader.json'), J('results', 'neg_sweep12', 'post_cal.json')]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, os.path.join('freqcfg', os.path.relpath(f, repo)).replace(os.sep, '/'))
print(f'{dst}: {len(files)} files, {os.path.getsize(dst) / 1e6:.2f} MB')
