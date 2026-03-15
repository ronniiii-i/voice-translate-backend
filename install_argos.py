import argostranslate.package

PAIRS = [
    ("en", "fr"), ("fr", "en"),
    ("en", "de"), ("de", "en"),
    ("en", "es"), ("es", "en"),
    ("en", "zh"), ("zh", "en"),
    ("fr", "de"), ("de", "fr"),
    ("fr", "es"), ("es", "fr"),
    ("de", "es"), ("es", "de"),
]

print("Updating Argos package index...")
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
available_map = {(p.from_code, p.to_code): p for p in available}

for src, tgt in PAIRS:
    pkg = available_map.get((src, tgt))
    if pkg:
        print(f"Installing {src} -> {tgt}...")
        argostranslate.package.install_from_path(pkg.download())
    else:
        print(f"Skipping {src} -> {tgt} (will pivot via English at runtime)")

print("Done.")