"""Manual staging preserves sources, resolves assets, and catches broken sections."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from examples import documentation as manual

class DocumentationTests(unittest.TestCase):
    def test_staging_rewrites_links_without_modifying_sources(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'docs/start').mkdir(parents=True);(root/'assets').mkdir()
            (root/'docs/index.md').write_text('# Welcome\n[Reference](README.md)\n[First](start/first.md#go)\n')
            (root/'docs/README.md').write_text('# Reference\n[Home](index.md)\n')
            page=root/'docs/start/first.md';text='# First\n[Root](../../README.md)\n![Picture](../../assets/image.png)\n```md\n[example](not-a-real-file.md)\n```\n';page.write_text(text)
            (root/'README.md').write_text('# Repository\n[Guide](docs/index.md)\n')
            (root/'assets/image.png').write_bytes(b'image fixture')
            with patch.multiple(manual,ROOT=root,SOURCE=root/'build/manual/source'):
                self.assertEqual(manual.prepare(),4)
                stage=manual.SOURCE
                self.assertIn('(reference-index.md)',(stage/'index.md').read_text())
                self.assertIn('(../project/README.md)',(stage/'start/first.md').read_text())
                self.assertIn('(../assets/image.png)',(stage/'start/first.md').read_text())
                self.assertTrue((stage/'assets/image.png').is_file())
                self.assertIn('(../index.md)',(stage/'project/README.md').read_text())
                self.assertEqual(page.read_text(),text)
                self.assertEqual(manual.prepare(),4) # Regeneration is deterministic.
                self.assertEqual(page.read_text(),text)

    def test_missing_and_outside_links_fail_before_replacing_staging(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'docs').mkdir();(root/'docs/README.md').write_text('# Reference');stage=root/'build/manual/source';stage.mkdir(parents=True)
            keep=stage/'keep';keep.write_text('prior build')
            with patch.multiple(manual,ROOT=root,SOURCE=stage):
                for target in ('missing.md','../../outside.md'):
                    (root/'docs/index.md').write_text('[Bad]('+target+')')
                    with self.assertRaises(ValueError):manual.prepare()
                    self.assertEqual(keep.read_text(),'prior build')

    def test_rendered_file_image_and_anchor_checks(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'assets/vendor/iframe-worker').mkdir(parents=True)
            (root/'assets/vendor/iframe-worker/shim.js').write_text('// fixture')
            (root/'search.json').write_text('{}');(root/'image.png').write_bytes(b'fixture')
            index=root/'index.html';index.write_text('<a href="topic.html#settings">Topic</a><img src="image.png">')
            (root/'topic.html').write_text('<h1 id="settings">Settings</h1>')
            self.assertEqual(manual.check_site(root),2)
            for bad in ('missing.png','topic.html#absent'):
                index.write_text('<a href="'+bad+'">Broken</a>')
                with self.assertRaisesRegex(ValueError,'Broken manual links'):manual.check_site(root)

    def test_identifying_home_paths_are_rejected_in_public_text(self):
        for prefix in ('/'+'home/', '/'+'Users/', 'C:'+'\\Users\\'):
            with self.subTest(prefix=prefix),self.assertRaisesRegex(ValueError,'Private home-directory path'):
                manual.validate_public_text(prefix+'example-user/project','public guide')
        manual.validate_public_text('assets/examples/manual-interior.csv','public guide')

    def test_local_offline_shim_and_current_navigation_sources(self):
        import tomllib
        root=Path(__file__).resolve().parents[1];config=tomllib.loads((root/'zensical.toml').read_text())['project']
        self.assertIn('offline',config['plugins'])
        self.assertFalse(config['theme']['font'])
        self.assertEqual(config['extra']['polyfills'],['assets/vendor/iframe-worker/shim.js'])
        self.assertTrue((root/config['extra']['polyfills'][0]).is_file())
        def walk(entries):
            for entry in entries:
                for title,value in entry.items():
                    if isinstance(value,list):yield from walk(value)
                    else:yield value
        for page in walk(config['nav']):
            source='README.md' if page=='reference-index.md' else page
            self.assertTrue((root/'docs'/source).is_file(),page)
