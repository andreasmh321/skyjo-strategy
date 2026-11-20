"""Rebuild the playable page from a checkpoint:
  python web/export.py            (writes web/weights.json; edit the checkpoint path inside if needed)
  python web/build_page.py        (writes web/skyjo_vs_agent.html)
  node web/test_core.js           (optional: checks the JS port against web/cases.json)
"""
import os
d = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(d, "page_template.html"), encoding="utf-8").read()
w = open(os.path.join(d, "weights.json"), encoding="utf-8").read()
core = open(os.path.join(d, "skyjo_core.js"), encoding="utf-8").read()
html = t.replace("/*WEIGHTS*/", w).replace("/*CORE*/", core)
open(os.path.join(d, "skyjo_vs_agent.html"), "w", encoding="utf-8").write(html)
print(f"wrote skyjo_vs_agent.html ({len(html) / 1e6:.1f} MB)")
