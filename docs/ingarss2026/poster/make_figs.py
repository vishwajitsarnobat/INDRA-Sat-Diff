"""QR code for the poster (links to the repository)."""
import os

import qrcode

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
qrcode.make("https://github.com/vishwajitsarnobat/INDRA-Sat-Diff", box_size=20, border=1).save(os.path.join(OUT, "qr_github.png"))
