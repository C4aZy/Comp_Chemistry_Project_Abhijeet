PES Analysis Tool – Research Lab Edition
A Professional, Zero-Setup Web App for Potential Energy Surface Analysis
PES Tool Screenshot
(Interactive plot • Annotated Matplotlib • Boltzmann populations • One-click reports)
The tool every computational chemist actually wants
Tired of copying data into Excel, manually finding minima, or writing one-off Python scripts every time you run a dihedral scan?
This is the complete, beautiful, ready-to-use web application you've been waiting for.
Drop in your scan → get publication-ready analysis in seconds.

Features

Smart File Upload
Supports real-world files:
Gaussian .log / .out
ORCA relaxed surface scan .out
Simple CSV or space-separated TXT
Drag & drop or click to upload

Manual Data Entry
Add, edit, or delete points directly in the browser with a clean table
Instant Interactive Plot
Powered by Chart.js – smooth, responsive, zoomable
One-Click Advanced Analysis
Everything you need for conformational studies:
Global and local minima
Transition states (maxima)
Energy barriers (kJ/mol and kcal/mol)
Cubic spline interpolation
Fourier series fitting (V0 + V1–V3 terms)
Full Boltzmann distribution at 298.15 K
Conformational populations (%) for every point and conformer
Entropy contribution from conformational mixing
Number of significant conformers (>1% population)

Beautiful Annotated Matplotlib Plot
Marks global minimum (★), local minima (▲), transition states (▼)
Energy values labeled directly on plot
Secondary histogram of energy distribution
High-resolution PNG (300 DPI)

Download Everything
Raw data as CSV
Interactive plot as PNG
Full scientific text report (ready to paste into papers or theses)
Annotated analysis plot

Dedicated Conformational Stability Panel
Clear summary of most/least stable conformers, stability range, and top populated structures
100% Offline • No Database • No Config
Pure Python + HTML. Works on your laptop forever.


How to Use (Seriously, That’s It)

Make sure you have Python 3.8+
Install dependencies (once):Bashpip install flask pandas numpy matplotlib scipy
Download the two files:
app.py
templates/index.html ← (create folder templates and put it inside)

Run:Bashpython app.py
Open your browser: http://localhost:5000

You’re done. Start analyzing.

Perfect For

Rotational barrier studies
Atropisomerism and axial chirality
Ring flips (cyclohexane, heterocycles)
Amide rotation, N–C(O) barriers
Organophosphorus, organometallic, and inorganic torsion scans
Teaching computational chemistry
Quick validation of scan convergence


Why This Tool Exists
Because every computational chemist has done this 100 times:
“Let me just copy the scan energies… find the minimum… calculate ΔG… make a plot in Origin… ugh.”
Now you don’t have to.
This tool is battle-tested on real research projects (2024–2025) involving biphenyls, BINOLs, ferrocenes, atropisomeric drugs, and more.
